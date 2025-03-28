# # Marilyn Braojos 
# # Mariam Misabishvili

import rclpy
from rclpy.node import Node  
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist, Vector3Stamped 
import math
import numpy as np

WAYPOINTS_FILE = "wayPoints.txt"
WAYPOINT_TOLERANCES = [0.1, 0.1, 0.1]  # Radius around each goal to stop
STOP_DURATIONS = [10, 2, 2]  # Stop time (seconds) for each goal

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')

        # creating publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.obs_sub = self.create_subscription(Vector3Stamped, '/detected_distance', self.lidar_callback, 10)

        # initialize current goal index
        self.goal_index = 0

        # Read list of waypoints from file
        self.waypoints = self.read_waypoints()
        self.current_position = (0.0, 0.0)
        self.yaw = 0.0  # orientation in radians
        self.odom_offset = None  # flag to trigger taring
        self.init_yaw = 0.0
        self.init_pos = None
        self.rotation_matrix = None
        self.init_x = 0.0
        self.init_y = 0.0
        
        # Timestamp when robot reaches current goal
        self.reached_time = None

        # Start time for global timeout (2 min 30 sec = 150s)
        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]

        # Timer for control loop (every 0.1s)
        self.timer = self.create_timer(0.1, self.controller_loop)
        
        self.obstacle_distance = None
        self.robot_footprint_radius = 0.25  # Adjust based on your robot
        self.safety_buffer = 0.05  # Additional buffer in meters
        
        self.avoiding_obstacle = False
        self.avoid_step = 0  # 0 = not avoiding, 1 = turning, 2 = moving forward
        self.avoid_start_time = None
        self.avoid_duration = 2.0  # seconds to turn or move forward (tune this!)

        self.avoid_start_position = None
        self.avoid_forward_distance = 0.4

    def read_waypoints(self):
        # Read waypoints from text file
        waypoints = []
        with open(WAYPOINTS_FILE, 'r') as f:
            for line in f:
                x, y = map(float, line.strip().split())
                waypoints.append((x, y))
        return waypoints

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        orientation = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        if self.odom_offset is None:
            self.odom_offset = True
            self.init_yaw = orientation
            self.init_pos = position

            # Rotation matrix to align with initial yaw
            self.rotation_matrix = np.array([
                [np.cos(self.init_yaw), np.sin(self.init_yaw)],
                [-np.sin(self.init_yaw), np.cos(self.init_yaw)]
            ])

            # Transform initial position
            self.init_x = self.rotation_matrix[0, 0] * position.x + self.rotation_matrix[0, 1] * position.y
            self.init_y = self.rotation_matrix[1, 0] * position.x + self.rotation_matrix[1, 1] * position.y

        # Transform current position
        transformed_x = self.rotation_matrix[0, 0] * position.x + self.rotation_matrix[0, 1] * position.y
        transformed_y = self.rotation_matrix[1, 0] * position.x + self.rotation_matrix[1, 1] * position.y

        self.current_position = (
            transformed_x - self.init_x,
            transformed_y - self.init_y
        )

        self.yaw = orientation - self.init_yaw

    def lidar_callback(self, msg: Vector3Stamped):
        self.obstacle_distance = msg.vector.z
        if self.obstacle_distance < 0.20 and not self.avoiding_obstacle:
            self.avoiding_obstacle = True # set avoiding obstacle to true 

    def controller_loop(self):
        if self.goal_index >= len(self.waypoints):
            self.cmd_pub.publish(Twist())  # All goals reached, stop the robot
            return

        now = self.get_clock().now().seconds_nanoseconds()[0]
        if now - self.start_time > 150:
            self.get_logger().info("Time limit exceeded!")
            self.cmd_pub.publish(Twist())  # Stop if time exceeds limit
            return
        
        if self.avoiding_obstacle:
            
            
            # Step 1: Turn 90 degrees to the left
            if self.avoid_step == 0:
                self.get_logger().info("⚠️ Obstacle detected. Starting avoidance.")
                self.avoid_start_yaw = self.yaw
                self.avoid_step = 1

            elif self.avoid_step == 1:
                cmd = Twist()
                # Normalize angle difference
                angle_diff = math.atan2(
                    math.sin(self.yaw - self.avoid_start_yaw),
                    math.cos(self.yaw - self.avoid_start_yaw)
                )
                angle_turned = abs(angle_diff)

                tolerance_rad = math.radians(15)
                target_angle = math.pi / 2  # 90 degrees in radians

                self.get_logger().info(f"🔄 Turning... {math.degrees(angle_turned)}° turned")
                self.get_logger().info(f"angle_diff {angle_diff} turned")

                if (target_angle - tolerance_rad) <= angle_turned <= (target_angle + tolerance_rad):
                    self.get_logger().info("↪️ Turn complete. Moving forward.")
                    self.cmd_pub.publish(Twist())
                    self.avoid_step = 2
                    self.avoid_start_position = self.current_position
                    self.get_logger().info(f"angle_diff {self.avoid_start_position}")
                else: 
                    
                    cmd.angular.z = 0.5
                    self.cmd_pub.publish(cmd)
                    print("rotating")
                    return 
                
            elif self.avoid_step == 2:

                cmd = Twist()
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                
                dx = self.current_position[0] - self.avoid_start_position[0]
                dy = self.current_position[1] - self.avoid_start_position[1]
                dist_moved = math.sqrt(dx**2 + dy**2)

                self.get_logger().info(f"dx {dx}")
                self.get_logger().info(f"dy {dy} ")
                self.get_logger().info(f"dist_moved {dist_moved}")

                if dist_moved >= self.avoid_forward_distance:
                    self.get_logger().info("✅ Avoidance complete. Resuming navigation.")
                    self.avoiding_obstacle = False
                else: 
                    #cmd = Twist()
                    # self.cmd_pub.publish(Twist())
                    # Move forward until we've moved 40 cm
                    cmd.linear.x = 0.25
                    # cmd.angular.z = 0.0
                    self.cmd_pub.publish(cmd)
                    
                    return

        # Get current goal point
        goal = self.waypoints[self.goal_index]
        dx = goal[0] - self.current_position[0]
        dy = goal[1] - self.current_position[1]
        distance = math.sqrt(dx**2 + dy**2)

        # Check if we're close enough to the goal
        if distance < WAYPOINT_TOLERANCES[self.goal_index]:
            if self.reached_time is None:
                self.reached_time = now  # Start wait timer
            elif now - self.reached_time >= STOP_DURATIONS[self.goal_index]:
                self.goal_index += 1  # Move to next goal
                self.reached_time = None  # Reset
            self.cmd_pub.publish(Twist())  # Hold position
            return

        # Combine goal direction with obstacle avoidance ejwieiwrwue
        total_dx = dx 
        total_dy = dy 

        # Compute angle and angle difference from robot orientation
        angle_to_goal = math.atan2(total_dy, total_dx)
        angle_diff = angle_to_goal - self.yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalize angle

        cmd = Twist()
        if abs(angle_diff) > 0.1:
            cmd.angular.z = 0.5 * angle_diff  # Turn toward goal
        else:
            cmd.linear.x = 0.15  # Move forward
            cmd.angular.z = 0.3 * angle_diff  # Minor steering

        self.cmd_pub.publish(cmd)  # Send command to robot

def main(args=None):
    rclpy.init(args=args)  # initialize ROS
    node = GoToGoal()  # create node
    rclpy.spin(node) 
    node.destroy_node()  # clean-up
    rclpy.shutdown()  # shutdown ROS

if __name__ == '__main__':
    main()