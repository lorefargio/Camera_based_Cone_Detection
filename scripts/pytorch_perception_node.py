import os
import sys
import time
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package is required to run PyTorch benchmarks.")
    sys.exit(1)

class PyTorchPerceptionNode(Node):
    def __init__(self, model_path, output_csv):
        super().__init__('pytorch_perception_node')
        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.output_csv = output_csv
        self.stats = []
        self.count = 0
        
        # Subscribe to the same camera image topic
        self.sub = self.create_subscription(
            Image, '/zed/zed_node/rgb/color/rect/image', self.image_callback, 10)
        self.get_logger().info(f"PyTorch Perception Node initialized with model {model_path}")
        
    def image_callback(self, msg):
        start_time = time.perf_counter()
        
        try:
            # Convert ROS Image to OpenCV Mat
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {str(e)}")
            return
            
        # Run inference
        results = self.model(cv_img, verbose=False)
        
        end_time = time.perf_counter()
        total_ms = (end_time - start_time) * 1000.0
        hz = 1000.0 / total_ms if total_ms > 0 else 0.0
        
        # Get timings in ms
        speed = results[0].speed
        preprocess_ms = speed.get('preprocess', 0.0)
        inference_ms = speed.get('inference', 0.0)
        postprocess_ms = speed.get('postprocess', 0.0)
        
        self.count += 1
        # Skip the first frame for warm-up metrics (like C++ node does)
        if self.count > 1:
            self.stats.append({
                'timestamp': msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec,
                'latency_ms': total_ms,
                'hz': hz,
                'detections': len(results[0].boxes) if results[0].boxes is not None else 0,
                'preprocess_ms': preprocess_ms,
                'inference_ms': inference_ms,
                'postprocess_ms': postprocess_ms
            })
            
            if self.count % 10 == 0:
                self.get_logger().info(f"Frame {self.count} | LATENCY: {total_ms:.2f} ms | FREQUENCY: {hz:.2f} Hz")
                
    def save_stats(self):
        if self.stats:
            df = pd.DataFrame(self.stats)
            df.to_csv(self.output_csv, index=False)
            self.get_logger().info(f"Saved PyTorch stats to {self.output_csv}")
        else:
            self.get_logger().warn("No stats to save.")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 pytorch_perception_node.py <model_path.pt> <output_csv>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    output_csv = sys.argv[2]
    
    rclpy.init()
    node = PyTorchPerceptionNode(model_path, output_csv)
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info("Shutting down PyTorch node...")
    finally:
        node.save_stats()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
