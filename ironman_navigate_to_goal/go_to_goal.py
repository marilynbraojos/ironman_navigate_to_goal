# Marilyn Braojos 
# Mariam Misabishvili

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist, Vector3Stamped 
from sensor_msgs.msg import CompressedImage, LaserScan
import math
import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image
from collections import Counter
from cv_bridge import CvBridge

LABEL_MAP = {
    0: "empty wall",
    1: "left",
    2: "right",
    3: "do not enter",
    4: "stop",
    5: "goal"
}

MODEL_PATH = "./sign_classifier.pkl"

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.obs_sub = self.create_subscription(Vector3Stamped, '/detected_distance', self.lidar_callback, 10)
        self.image_sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10)

        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, lidar_qos)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.model = self.initialize_model(MODEL_PATH)

        self.yaw = 0.0
        self.init_yaw = 0.0
        self.init_x = 0.0
        self.init_y = 0.0
        self.avoid_step = 0
        self.current_position = (0.0, 0.0)
        self.odom_offset = None
        self.init_pos = None
        self.rotation_matrix = None
        self.obstacle_distance = None
        self.avoid_start_time = None
        self.avoiding_obstacle = False

        self.classification_rounds = []
        self.classification_step = 0

        self.latest_scan = None
        self.turn_direction = None

        self.timer = self.create_timer(0.1, self.controller_loop)

    def initialize_model(self, model_path):
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def predict(self, model, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        image = test_transform(image).unsqueeze(0)
        image = image.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        orientation = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        if self.odom_offset is None:
            self.odom_offset = True
            self.init_yaw = orientation
            self.init_pos = position
            self.rotation_matrix = np.array([
                [np.cos(self.init_yaw), np.sin(self.init_yaw)],
                [-np.sin(self.init_yaw), np.cos(self.init_yaw)]
            ])
            self.init_x = self.rotation_matrix[0, 0] * position.x + self.rotation_matrix[0, 1] * position.y
            self.init_y = self.rotation_matrix[1, 0] * position.x + self.rotation_matrix[1, 1] * position.y

        transformed_x = self.rotation_matrix[0, 0] * position.x + self.rotation_matrix[0, 1] * position.y
        transformed_y = self.rotation_matrix[1, 0] * position.x + self.rotation_matrix[1, 1] * position.y

        self.current_position = (transformed_x - self.init_x, transformed_y - self.init_y)
        self.yaw = orientation - self.init_yaw

    def lidar_callback(self, msg: Vector3Stamped):
        self.obstacle_distance = msg.vector.z
        if self.obstacle_distance < 0.50 and not self.avoiding_obstacle:
            self.avoiding_obstacle = True
            self.avoid_step = 0

    def controller_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.avoiding_obstacle:
            if self.avoid_step == 0:
                self.get_logger().info("🚨 Obstacle detected. Stopping and preparing to classify.")
                self.cmd_pub.publish(Twist())
                self.avoid_step = 1
                return

            elif self.avoid_step == 1:
                if self.classification_step < 5:
                    self.get_logger().info(f"📸 Classification round {self.classification_step + 1}...")
                    if self.latest_frame is None:
                        self.get_logger().warn("🛑 No image available.")
                        return

                    votes = [self.predict(self.model, self.latest_frame) for _ in range(11)]
                    round_vote = Counter(votes).most_common(1)[0][0]
                    self.classification_rounds.append(round_vote)
                    self.classification_step += 1
                    self.get_logger().info(f"✅ Round {self.classification_step} vote: {LABEL_MAP[round_vote]}")

                    cmd = Twist()
                    if self.classification_step == 1:
                        cmd.linear.x = -0.1
                    elif self.classification_step == 2:
                        cmd.linear.x = 0.1
                    self.cmd_pub.publish(cmd)
                    return

                else:
                    final_decision = Counter(self.classification_rounds).most_common(1)[0][0]
                    self.get_logger().info(f"Final decision: {LABEL_MAP[final_decision]}")
                    self.classification_step = 0
                    self.classification_rounds = []
                    self.avoid_start_time = now
                    self.turn_direction = None
                    self.avoid_step = 2 + final_decision
                    return

            elif self.avoid_step == 2: 
                if self.avoid_start_time is None:
                    self.avoid_start_time = now
                    self.turn_direction = None

                elapsed = now - self.avoid_start_time

                if elapsed < 3:
                    self.get_logger().info("🔙 Backing up from empty wall...")
                    cmd = Twist()
                    cmd.linear.x = -0.4
                    self.cmd_pub.publish(cmd)
                    return

                elif elapsed < 5:
                    if self.turn_direction is None:
                        if not self.latest_scan:
                            self.get_logger().warn("⚠️ LIDAR scan unavailable.")
                            return

                        angle_min = self.latest_scan.angle_min
                        angle_inc = self.latest_scan.angle_increment
                        ranges = self.latest_scan.ranges
                        num_ranges = len(ranges)

                        max_left = 0.0
                        max_right = 0.0

                        for i in range(num_ranges):
                            angle = angle_min + i * angle_inc
                            angle_deg = math.degrees(angle % (2 * math.pi))

                            if 0 <= angle_deg <= 31:
                                if 0.0 < ranges[i] < float('inf'):
                                    max_right = max(max_right, ranges[i])
                            elif 329 <= angle_deg <= 360:
                                if 0.0 < ranges[i] < float('inf'):
                                    max_left = max(max_left, ranges[i])

                        self.turn_direction = "left" if max_left > max_right else "right"
                        self.get_logger().info(f"Chose to turn {self.turn_direction} (left={max_left:.2f}m, right={max_right:.2f}m)")

                    cmd = Twist()
                    cmd.angular.z = -0.3 if self.turn_direction == "left" else 0.3
                    self.get_logger().info(f"🔄 Turning {self.turn_direction}")
                    self.cmd_pub.publish(cmd)
                    return

                elif elapsed < 5.0:
                    self.get_logger().info("➡️ Continuing forward")
                    cmd = Twist()
                    cmd.linear.x = 0.1
                    self.cmd_pub.publish(cmd)
                    return

                else:
                    self._reset()
                    self.get_logger().info("✅ Resuming navigation.")
                    return

            elif self.avoid_step in [3, 4, 5, 6]:
                angular_values = {3: 0.8, 4: -0.8, 5: -1.6, 6: 1.6}
                self._turn_and_move(now, angular_values[self.avoid_step])

            elif self.avoid_step == 7:
                self.get_logger().info("🎉 GOAL reached. Robot will stop permanently.")
                self.cmd_pub.publish(Twist())
                self.avoiding_obstacle = True
                return

        else:
            cmd = Twist()
            cmd.linear.x = 0.1
            self.cmd_pub.publish(cmd)
            self.get_logger().info("🚗 Going forward")

    def _turn_and_move(self, now, angular_z):
        if self.avoid_start_time is None:
            self.avoid_start_time = now

        if now - self.avoid_start_time < 2.0:
            cmd = Twist()
            cmd.angular.z = angular_z
            self.cmd_pub.publish(cmd)
        elif now - self.avoid_start_time < 4.0:
            cmd = Twist()
            cmd.linear.x = 0.1
            self.cmd_pub.publish(cmd)
        else:
            self._reset()

    def _reset(self):
        self.avoiding_obstacle = False
        self.avoid_step = 0
        self.avoid_start_time = None
        self.turn_direction = None
        self.cmd_pub.publish(Twist())
        self.get_logger().info("✅ Reset. Resuming navigation.")

def main(args=None):
    rclpy.init(args=args)
    node = GoToGoal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()