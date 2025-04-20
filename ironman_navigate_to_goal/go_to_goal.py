# Marilyn Braojos 
# Mariam Misabishvili

import rclpy
from rclpy.node import Node  
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist, Vector3Stamped 
import math
import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image
from collections import Counter
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

LABEL_MAP = {
    0: "empty wall",
    1: "left",
    2: "right",
    3: "do not enter",
    4: "stop",
    5: "goal"
}

MODEL_PATH = "./sign_classifier.pkl"  # Path to your model

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.obs_sub = self.create_subscription(Vector3Stamped, '/detected_distance', self.lidar_callback, 10)
        self.image_sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10)

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
        if self.obstacle_distance < 0.45 and not self.avoiding_obstacle:
            self.avoiding_obstacle = True
            self.avoid_step = 0

    def controller_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.avoiding_obstacle:
            if self.avoid_step == 0:
                self.get_logger().info("🚨 Obstacle detected. Stopping and preparing to classify...")
                self.cmd_pub.publish(Twist())  # Stop
                self.avoid_step = 1
                return

            elif self.avoid_step == 1:
                if self.classification_step < 5:  # CHANGED from 3 to 5 rounds
                    self.get_logger().info(f"📸 Classification round {self.classification_step + 1}...")
                    if self.latest_frame is None:
                        self.get_logger().warn("🛑 No image available.")
                        return

                    votes = []
                    for _ in range(11):
                        pred = self.predict(self.model, self.latest_frame)
                        votes.append(pred)

                    round_vote = Counter(votes).most_common(1)[0][0]
                    self.classification_rounds.append(round_vote)
                    self.classification_step += 1
                    self.get_logger().info(f"✅ Round {self.classification_step} vote: {LABEL_MAP[round_vote]}")

                    cmd = Twist()
                    if self.classification_step == 1:
                        cmd.linear.x = -0.1  # Back up
                    elif self.classification_step == 2:
                        cmd.linear.x = 0.1   # Move forward
                    self.cmd_pub.publish(cmd)
                    return

                else:
                    final_decision = Counter(self.classification_rounds).most_common(1)[0][0]
                    self.get_logger().info(f"🧠 Final decision: {LABEL_MAP[final_decision]}")
                    self.classification_step = 0
                    self.classification_rounds = []
                    self.avoid_start_time = now

                    if final_decision == 0:      # empty wall
                        self.avoid_step = 2
                    elif final_decision == 1:    # turn left
                        self.avoid_step = 3
                    elif final_decision == 2:    # turn right
                        self.avoid_step = 4
                    elif final_decision == 3:    # do not enter
                        self.avoid_step = 5
                    elif final_decision == 4:    # stop
                        self.avoid_step = 6
                    elif final_decision == 5:    # goal
                        self.avoid_step = 7
                    return

            elif self.avoid_step == 2:  # EMPTY WALL → faster spin
                self.get_logger().info("🧱 Empty wall detected. Rotating to look for new sign quickly...")
                cmd = Twist()
                cmd.angular.z = 0.8  # Increased spin speed
                self.cmd_pub.publish(cmd)

                if self.obstacle_distance is not None and self.obstacle_distance < 0.45:
                    self.get_logger().info("👀 New obstacle spotted! Restarting classification...")
                    self.avoid_step = 1
                    self.classification_step = 0
                    self.classification_rounds = []
                    self.cmd_pub.publish(Twist())  # Stop before re-classification
                return

            elif self.avoid_step == 3:  # TURN LEFT
                if now - self.avoid_start_time < 2.0:
                    cmd = Twist()
                    cmd.angular.z = 0.8
                    self.cmd_pub.publish(cmd)
                elif now - self.avoid_start_time < 4.0:
                    cmd = Twist()
                    cmd.linear.x = 0.1
                    self.cmd_pub.publish(cmd)
                else:
                    self._reset()
                return

            elif self.avoid_step == 4:  # TURN RIGHT
                if now - self.avoid_start_time < 2.0:
                    cmd = Twist()
                    cmd.angular.z = -0.8
                    self.cmd_pub.publish(cmd)
                elif now - self.avoid_start_time < 4.0:
                    cmd = Twist()
                    cmd.linear.x = 0.1
                    self.cmd_pub.publish(cmd)
                else:
                    self._reset()
                return

            elif self.avoid_step == 5:  # DO NOT ENTER = 180° turn
                if now - self.avoid_start_time < 3.0:
                    cmd = Twist()
                    cmd.angular.z = -1.2
                    self.cmd_pub.publish(cmd)
                elif now - self.avoid_start_time < 5.0:
                    cmd = Twist()
                    cmd.linear.x = 0.1
                    self.cmd_pub.publish(cmd)
                else:
                    self._reset()
                return

            elif self.avoid_step == 6:  # STOP SIGN = 180° turn
                if now - self.avoid_start_time < 3.0:
                    cmd = Twist()
                    cmd.angular.z = 1.2
                    self.cmd_pub.publish(cmd)
                elif now - self.avoid_start_time < 5.0:
                    cmd = Twist()
                    cmd.linear.x = 0.1
                    self.cmd_pub.publish(cmd)
                else:
                    self._reset()
                return

            elif self.avoid_step == 7:  # GOAL = stop forever
                self.get_logger().info("🎉 GOAL reached. Robot will now stop permanently.")
                self.cmd_pub.publish(Twist())
                self.avoiding_obstacle = True  # prevent re-entering controller loop
                return

        else:
            cmd = Twist()
            cmd.linear.x = 0.1
            self.cmd_pub.publish(cmd)
            self.get_logger().info("🚗 Going forward")

    def _reset(self):
        self.avoiding_obstacle = False
        self.avoid_step = 0
        self.avoid_start_time = None
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


# # Marilyn Braojos 
# # Mariam Misabishvili

# import rclpy
# from rclpy.node import Node  
# from nav_msgs.msg import Odometry 
# from geometry_msgs.msg import Twist, Vector3Stamped 
# import math
# import numpy as np
# import torch
# import cv2
# from torchvision import transforms
# from PIL import Image
# from collections import Counter
# from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge


# LABEL_MAP = {
#     0: "empty wall",
#     1: "left",
#     2: "right",
#     3: "do not enter",
#     4: "stop",
#     5: "goal"
# }

# MODEL_PATH = "./sign_classifier.pkl"  # <-- CHANGE THIS

# class GoToGoal(Node):
#     def __init__(self):
#         super().__init__('go_to_goal')

#         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
#         self.obs_sub = self.create_subscription(Vector3Stamped, '/detected_distance', self.lidar_callback, 10)

#         self.bridge = CvBridge()
#         self.latest_frame = None  # stores most recent camera frame

#         self.image_sub = self.create_subscription(
#             CompressedImage,
#             '/image_raw/compressed',  # or replace with your correct topic
#             self.image_callback,
#             10
# )

#         self.yaw = 0.0
#         self.init_yaw = 0.0
#         self.init_x = 0.0
#         self.init_y = 0.0
#         self.avoid_step = 0
#         self.current_position = (0.0, 0.0)
#         self.odom_offset = None
#         self.init_pos = None
#         self.rotation_matrix = None
#         self.obstacle_distance = None
#         self.avoid_start_time = None
#         self.avoiding_obstacle = False

#         self.model = self.initialize_model(MODEL_PATH)

#         self.timer = self.create_timer(0.1, self.controller_loop)

#     def initialize_model(self, model_path):
#         model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
#         model.eval()
#         return model
    
#     def image_callback(self, msg):
#         try:
#             self.latest_frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
#         except Exception as e:
#             self.get_logger().error(f"Failed to convert image: {e}")


#     def predict(self, model, image):
#         if isinstance(image, np.ndarray):
#             image = Image.fromarray(image).convert("RGB")

#         test_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                  std=[0.229, 0.224, 0.225])
#         ])

#         image = test_transform(image).unsqueeze(0)
#         image = image.to(next(model.parameters()).device)

#         with torch.no_grad():
#             outputs = model(image)
#             _, predicted = torch.max(outputs, 1)

#         return predicted.item()

#     def classify_obstacle(self):

#         if self.latest_frame is None:
#             self.get_logger().warn("🛑 No image available for classification.")
#             return

#         self.cmd_pub.publish(Twist())  # stop robot
#         votes = []
#         for _ in range(21):
#             pred = self.predict(self.model, self.latest_frame)
#             votes.append(pred)

#         majority_vote = Counter(votes).most_common(1)[0][0]
#         label = LABEL_MAP.get(majority_vote, "Unknown")
#         self.get_logger().info(f"🧠 Classification Result: {label}")
#         self.get_logger().info(f"🧠 Votes: {votes}")
#         return majority_vote

#     def odom_callback(self, msg):
#         position = msg.pose.pose.position
#         q = msg.pose.pose.orientation
#         orientation = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

#         if self.odom_offset is None:
#             self.odom_offset = True
#             self.init_yaw = orientation
#             self.init_pos = position
#             self.rotation_matrix = np.array([
#                 [np.cos(self.init_yaw), np.sin(self.init_yaw)],
#                 [-np.sin(self.init_yaw), np.cos(self.init_yaw)]
#             ])
#             self.init_x = self.rotation_matrix[0, 0] * position.x + self.rotation_matrix[0, 1] * position.y
#             self.init_y = self.rotation_matrix[1, 0] * position.x + self.rotation_matrix[1, 1] * position.y

#         transformed_x = self.rotation_matrix[0, 0] * position.x + self.rotation_matrix[0, 1] * position.y
#         transformed_y = self.rotation_matrix[1, 0] * position.x + self.rotation_matrix[1, 1] * position.y

#         self.current_position = (transformed_x - self.init_x, transformed_y - self.init_y)
#         self.yaw = orientation - self.init_yaw

#     def lidar_callback(self, msg: Vector3Stamped):
#         self.obstacle_distance = msg.vector.z
#         if self.obstacle_distance < 0.45 and not self.avoiding_obstacle:
#             self.avoiding_obstacle = True
#             self.avoid_step = 0

#     def controller_loop(self):
#         now = self.get_clock().now().nanoseconds / 1e9

#         if self.avoiding_obstacle:
#             if self.avoid_step == 0:
#                 self.get_logger().info("🚨 Obstacle detected. Stopping and classifying...")
#                 self.cmd_pub.publish(Twist())  # stop first
#                 # self.classify_obstacle()
#                 self.avoid_step = 1
#                 return
            
#             elif self.avoid_step == 1:
#                 # if self.obstacle_distance is not None and self.obstacle_distance < 0.40:
#                 self.get_logger().info("🧠 Running classification...")
#                 class_dec = self.classify_obstacle()  # 🔧 classify while robot is still
                
#                 if class_dec == 0: 
#                     self.avoid_step = 4
#                 elif class_dec == 1: 
#                     self.avoid_step = 5
#                 elif class_dec == 2: 
#                     self.avoid_step = 6
#                 elif class_dec == 3: 
#                     self.avoid_step = 7
#                 elif class_dec == 4: 
#                     self.avoid_step = 8
#                 elif class_dec == 5: 
#                     self.avoid_step = 9
#                 else: 
#                     self.avoid_step = 2
#                 return
            
#             ############################################## step 2 
#             elif self.avoid_step == 2:
#                 if self.obstacle_distance is not None and self.obstacle_distance < 0.40:
#                     cmd = Twist()
#                     cmd.linear.x = -0.1
#                     self.cmd_pub.publish(cmd)
#                     return
#                 else:
#                     self.get_logger().info("🔙 Backup complete. Preparing to left.")
#                     self.avoid_step = 3
#                     self.avoid_start_time = now
#                     return

#             ############################################## step 3
#             elif self.avoid_step == 3:
#                 if self.avoid_start_time is None:
#                     self.avoid_start_time = now

#                 if now - self.avoid_start_time < 1.5:
#                     cmd = Twist()
#                     cmd.angular.z = 0.6
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("↩️ Turning left")
#                     return
#                 elif now - self.avoid_start_time < 5.0:
#                     cmd = Twist()
#                     cmd.linear.x = 0.1
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("➡️ Moving forward")
#                     return
#                 else:
#                     self.get_logger().info("✅ Avoidance maneuver complete.")
#                     self.avoiding_obstacle = False
#                     self.avoid_step = 0
#                     self.avoid_start_time = None
#                     self.cmd_pub.publish(Twist())
#                     return
                
#             ############################################## step 4 - EMPTY WALL
                
#             elif self.avoid_step == 4:
#                 if self.obstacle_distance is not None and self.obstacle_distance < 0.40:
#                     cmd = Twist()
#                     cmd.linear.x = -0.1
#                     self.cmd_pub.publish(cmd)
#                     return
#                 else:
#                     self.get_logger().info("🔙 Backup complete. Detected an EMPTY WALL")
#                     self.avoid_step = 3
#                     self.avoid_start_time = now
#                     return
                
                
#             ############################################## step 5 - TURN LEFT
#             elif self.avoid_step == 5:
#                 if self.obstacle_distance is not None and self.obstacle_distance < 0.40:
#                     cmd = Twist()
#                     cmd.linear.x = -0.1
#                     self.cmd_pub.publish(cmd)
#                     return
#                 else:
#                     self.get_logger().info("🔙 Backup complete. Detected TURN LEFT")
#                     self.avoid_step = 10
#                     self.avoid_start_time = now
#                     return
                

#             elif self.avoid_step == 10:
#                 if self.avoid_start_time is None:
#                     self.avoid_start_time = now

#                 if now - self.avoid_start_time < 2.0:
#                     cmd = Twist()
#                     cmd.angular.z = 0.8
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("↩️ Turning left")
#                     return
#                 elif now - self.avoid_start_time < 5.0:
#                     cmd = Twist()
#                     cmd.linear.x = 0.1
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("➡️ Moving forward")
#                     return
#                 else:
#                     self.get_logger().info("✅ Avoidance maneuver complete.")
#                     self.avoiding_obstacle = False
#                     self.avoid_step = 0
#                     self.avoid_start_time = None
#                     self.cmd_pub.publish(Twist())
#                     return
#             ############################################## step 6 - TURN RIGHT
#             elif self.avoid_step == 6:
#                 if self.obstacle_distance is not None and self.obstacle_distance < 0.40:
#                     cmd = Twist()
#                     cmd.linear.x = -0.1
#                     self.cmd_pub.publish(cmd)
#                     return
#                 else:
#                     self.get_logger().info("🔙 Backup complete. Detected TURN RIGHT")
#                     self.avoid_step = 11
#                     self.avoid_start_time = now
#                     return
                
#             elif self.avoid_step == 11:
#                 if self.avoid_start_time is None:
#                     self.avoid_start_time = now

#                 if now - self.avoid_start_time < 2.0:
#                     cmd = Twist()
#                     cmd.angular.z = -0.8
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("↩️ Turning left")
#                     return
#                 elif now - self.avoid_start_time < 1.0:
#                     cmd = Twist()
#                     cmd.linear.x = 0.1
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("➡️ Moving forward")
#                     return
#                 else:
#                     self.get_logger().info("✅ Avoidance maneuver complete.")
#                     self.avoiding_obstacle = False
#                     self.avoid_step = 0
#                     self.avoid_start_time = None
#                     self.cmd_pub.publish(Twist())
#                     return
#             ############################################## step 7 - TURN AROUND
                
#             elif self.avoid_step == 7:
#                 if self.obstacle_distance is not None and self.obstacle_distance < 0.40:
#                     cmd = Twist()
#                     cmd.linear.x = -0.1
#                     self.cmd_pub.publish(cmd)
#                     return
#                 else:
#                     self.get_logger().info("🔙 Backup complete. Detected DO NOT ENTER")
#                     self.avoid_step = 12
#                     self.avoid_start_time = now
#                     return
                

#             elif self.avoid_step == 12:
#                 if self.avoid_start_time is None:
#                     self.avoid_start_time = now

#                 if now - self.avoid_start_time < 5.0:
#                     cmd = Twist()
#                     cmd.angular.z = -1.2
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("↩️ Turning left")
#                     return
#                 elif now - self.avoid_start_time < 1.0:
#                     cmd = Twist()
#                     cmd.linear.x = 0.1
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("➡️ Moving forward")
#                     return
#                 else:
#                     self.get_logger().info("✅ Avoidance maneuver complete.")
#                     self.avoiding_obstacle = False
#                     self.avoid_step = 0
#                     self.avoid_start_time = None
#                     self.cmd_pub.publish(Twist())
#                     return
#             ############################################## step 8 - TURN AROUND
                
#             elif self.avoid_step == 8:
#                 if self.obstacle_distance is not None and self.obstacle_distance < 0.40:
#                     cmd = Twist()
#                     cmd.linear.x = -0.1
#                     self.cmd_pub.publish(cmd)
#                     return
#                 else:
#                     self.get_logger().info("🔙 Backup complete. Detected STOP SIGN")
#                     self.avoid_step = 13
#                     self.avoid_start_time = now
#                     return
                
#             elif self.avoid_step == 13:
#                 if self.avoid_start_time is None:
#                     self.avoid_start_time = now

#                 if now - self.avoid_start_time < 2.6:
#                     cmd = Twist()
#                     cmd.angular.z = -1.2
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("↩️ Turning left")
#                     return
#                 elif now - self.avoid_start_time < 5.0:
#                     cmd = Twist()
#                     cmd.linear.x = 0.1
#                     self.cmd_pub.publish(cmd)
#                     self.get_logger().info("➡️ Moving forward")
#                     return
#                 else:
#                     self.get_logger().info("✅ Avoidance maneuver complete.")
#                     self.avoiding_obstacle = False
#                     self.avoid_step = 0
#                     self.avoid_start_time = None
#                     self.cmd_pub.publish(Twist())
#                     return
                
#             ############################################## step 9 - STOP 
#             elif self.avoid_step == 9:
#                 if self.obstacle_distance is not None and self.obstacle_distance < 0.40:
#                     cmd = Twist()
#                     cmd.linear.x = -0.1
#                     self.cmd_pub.publish(cmd)
#                     return
#                 else:
#                     self.get_logger().info("🔙 Backup complete. Detected the GOAL")
#                     self.avoid_step = 14
#                     self.avoid_start_time = now
#                     return
                
#             elif self.avoid_step == 14:
#                 self.get_logger().info("GOAL detected. Finished maze..")
#                 self.cmd_pub.publish(Twist())  # stop first
#                 return

#         else:
#             cmd = Twist()
#             cmd.linear.x = 0.1
#             self.cmd_pub.publish(cmd)
#             self.get_logger().info("🚗 Going forward")

# def main(args=None):
#     rclpy.init(args=args)
#     node = GoToGoal()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()