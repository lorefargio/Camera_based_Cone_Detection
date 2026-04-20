#include <rclcpp/rclcpp.hpp>
#include "zed_fusion_perception/zed_perception_node.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ZedPerceptionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
