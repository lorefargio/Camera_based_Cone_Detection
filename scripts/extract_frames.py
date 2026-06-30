import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FrameExtractor(Node):
    def __init__(self, output_dir, max_frames=200):
        super().__init__('frame_extractor')
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.count = 0
        os.makedirs(output_dir, exist_ok=True)
        self.sub = self.create_subscription(
            Image, '/zed/zed_node/rgb/color/rect/image', self.callback, 10)
        self.get_logger().info(f"Extracting up to {max_frames} frames to {output_dir}...")

    def callback(self, msg):
        if self.count >= self.max_frames:
            self.get_logger().info("Finished extracting frames. Exiting.")
            raise SystemExit
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            filename = os.path.join(self.output_dir, f"frame_{self.count:05d}.jpg")
            cv2.imwrite(filename, cv_img)
            self.count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to save frame: {str(e)}")

def main():
    rclpy.init()
    node = FrameExtractor("extracted_frames")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
