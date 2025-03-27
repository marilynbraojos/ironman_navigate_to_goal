# # Marilyn Braojos 
# # Mariam Misabishvili

import rclpy
from rclpy.node import Node  
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist, Vector3Stamped  # Velocity command & obstacle vector
import math
import numpy as np

WAYPOINTS_FILE = "wayPoints.txt"
WAYPOINT_TOLERANCES = [0.1, 0.15, 0.2]  # Radius around each goal to stop
STOP_DURATIONS = [10, 10, 10]  # Stop time (seconds) for each goal

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')  # Name of the node

        # publisher for robot velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # subscribe to robot odometry for position and orientation
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # subscribe to obstacle direction vector
        self.obs_sub = self.create_subscription(Vector3Stamped, '/detected_distance', self.lidar_callback, 10)





        # Initialize current goal index
        self.goal_index = 0

        # Read list of waypoints from file
        self.waypoints = self.read_waypoints()



        # robot's position and orientation
        self.current_position = (0.0, 0.0)


        self.yaw = 0.0  # Orientation in radians
        self.odom_offset = None  # Flag to trigger taring
        self.init_yaw = 0.0
        self.init_pos = None
        self.rotation_matrix = None
        self.init_x = 0.0
        self.init_y = 0.0
        # Obstacle vector
        self.obstacle_vector = None

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

        self.avoiding_obstacle = False
        self.avoid_step = 0
        self.avoid_start_time = None
        self.avoid_duration = 2.0



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
    # def obstacle_callback(self, msg):
        # Update the obstacle vector
        # self.obstacle_vector = (msg.vector.x, msg.vector.y)

    def lidar_callback(self, msg: Vector3Stamped):
        self.obstacle_distance = msg.vector.x
        if not self.avoiding_obstacle and self.obstacle_distance < 0.15:
            self.get_logger().warn("⚠️ Obstacle detected — starting avoidance!")
            self.avoiding_obstacle = True
            self.avoid_step = 1
            self.avoid_start_time = self.get_clock().now().seconds_nanoseconds()[0]


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
            cmd = Twist()
            if self.avoid_step == 1:
                cmd.angular.z = 0.5
                cmd.linear.x = 0.0
                if now - self.avoid_start_time >= 2:
                    self.avoid_step = 2
                    self.avoid_start_time = now
                    self.get_logger().info("✅ Finished turning. Now moving forward.")
            elif self.avoid_step == 2:
                cmd.linear.x = 0.15
                cmd.angular.z = 0.0
                if now - self.avoid_start_time >= 2:
                    self.avoiding_obstacle = False
                    self.avoid_step = 0
                    self.get_logger().info("✅ Avoidance complete. Resuming waypoint tracking.")
            self.cmd_pub.publish(cmd)
            return

        
            # Obstacle avoidance: Stop if too close
        if self.obstacle_distance is not None:
            min_safe_distance = self.robot_footprint_radius + self.safety_buffer
            if self.obstacle_distance < min_safe_distance:
                self.get_logger().warn("⚠️ Obstacle detected too close — stopping.")
                self.cmd_pub.publish(Twist())  # Stop the robot
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

        # # Obstacle avoidance using a simple repulsive vector
        # avoid_dx = 0.0
        # avoid_dy = 0.0
        # if self.obstacle_vector:
        #     ox, oy = self.obstacle_vector
        #     obs_dist = math.sqrt(ox**2 + oy**2)
        #     if obs_dist < 0.5:
        #         # Push away from obstacle (perpendicular vector)
        #         avoid_dx = -oy
        #         avoid_dy = ox

        # Combine goal direction with obstacle avoidance
        total_dx = dx #+ avoid_dx
        total_dy = dy #+ avoid_dy

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
    rclpy.init(args=args)  # Initialize ROS
    node = GoToGoal()  # Create the node
    rclpy.spin(node)  # Keep it running
    node.destroy_node()  # Cleanup
    rclpy.shutdown()  # Shutdown ROS

if __name__ == '__main__':
    main()

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, Point
# from nav_msgs.msg import Odometry
# from transforms3d.euler import quat2euler
# import math
# import time

# class VelocityController(Node):
#     def __init__(self):        
#         super().__init__('velocity_publisher')

#         # # Subscribe to the object distance topic
#         # self.distance_subscriber = self.create_subscription(
#         #     Point,
#         #     'detected_distance',
#         #     self.pixel_callback, 
#         #     10)
#         # self.distance_subscriber 

#         self.odom_subscriber = self.create_subscription(
#             Odometry,
#             '/odom',
#             self.odom_callback,
#             10
#         )

#         # Publisher for velocity commands
#         self._vel_publish = self.create_publisher(Twist, '/cmd_vel', 10)
        
#         # PID gains for angular control
#         self.Kp = 0.005    # Proportional gain [tuned]
#         self.Ki = 0.001    # Integral gain [tuned]
#         self.Kd = 0.001    # Derivative gain [tuned]

#         self.max_angular_speed = 1.0 
#         self.dead_zone = 20  # pix deadband

#         self.dead_zone_linear = .05  # distance deadband

#         # Linear control parameters
#         self.linear_Kp = 0.5 # Proportional gain for linear velocity
#         self.linear_Ki = 0.001 # integral gain for linear velocity
#         self.linear_Kd = 0.001 # derivative gain for linear velocity
        
#         self.target_distance = 0.4  # [m]
#         self.max_linear_speed = 0.3 # [should be +/- .1 m/s]

#         # Initialize error tracking for angular PID
#         self.last_error = 0.0
#         self.integral = 0.0
#         self.last_update_time = self.get_clock().now()

#         # Initialize error tracking for linear PID
#         self.last_linear_error = 0.0
#         self.linear_integral = 0.0
#         self.last_linear_update_time = self.get_clock().now()

#         # Timeout mechanism.
#         self.last_msg_time = self.get_clock().now()  # last received message time
#         self.timeout_duration = 1.0  # time [s] before stopping motion
#         # self.timer = self.create_timer(0.5, self.check_timeout)

#         # odom stuff 

#         self.goal_x = 0.0
#         self.goal_y = 1.0
#         self.goal_tolerance = 0.1  # 10 cm
        

#         self.goal_reached_time = None
#         self.wait_duration = 10.0  # 
#         self.timer = self.create_timer(0.1, self.control_loop)
#         self.current_pose = None

#     def odom_callback(self, msg: Odometry):
#         self.current_pose = msg.pose.pose
#         self.get_logger().info(f"Current pose: x={self.current_pose.position.x:.2f}, y={self.current_pose.position.y:.2f}")

#     def control_loop(self):
#         if self.current_pose is None:
#             return

#         # Get current position
#         x = self.current_pose.position.x
#         y = self.current_pose.position.y

#         # Get current orientation (yaw)
#         orientation_q = self.current_pose.orientation
#         (yaw, _, _) = quat2euler([
#             orientation_q.w,  # Note the order is different here!
#             orientation_q.x,
#             orientation_q.y,
#             orientation_q.z
#         ])

#         # Compute error to goal
#         dx = self.goal_x - x
#         dy = self.goal_y - y
#         distance = math.hypot(dx, dy)
#         angle_to_goal = math.atan2(dy, dx)
#         angle_error = angle_to_goal - yaw
#         angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))  # Normalize to [-pi, pi]

#         twist = Twist()

#         # If within tolerance, start wait timer
#         if distance <= self.goal_tolerance:
#             if self.goal_reached_time is None:
#                 self.goal_reached_time = self.get_clock().now()
#                 self.get_logger().info("Goal reached! Waiting...")

#             elapsed = (self.get_clock().now() - self.goal_reached_time).nanoseconds / 1e9
#             if elapsed >= self.wait_duration:
#                 self.get_logger().info("Wait complete. Stopping.")
#                 twist.linear.x = 0.0
#                 twist.angular.z = 0.0
#             else:
#                 twist.linear.x = 0.0
#                 twist.angular.z = 0.0  # keep still
#         else:
#             # Reset timer if we moved away
#             self.goal_reached_time = None

#             # Simple proportional controller
#             twist.linear.x = min(0.2, 0.5 * distance)
#             twist.angular.z = 1.5 * angle_error

#         self._vel_publish.publish(twist)

#     # def pixel_callback(self, msg: Point):
#     #     current_time = self.get_clock().now()
#     #     self.last_msg_time = current_time  # last received message time

#     #     pix_error = msg.x

#     #     distance = msg.y
#     #     distance_error = distance - self.target_distance
        
#     #     twist = Twist()

#     #     dt = (current_time-self.last_update_time).nanoseconds / 1e9 # dt [s]
        
#     #     # compute PID if the error > the dead zone
#     #     if (abs(pix_error) > self.dead_zone or abs(distance_error) > self.dead_zone_linear) and dt > 0.0:
            
#     #         # integral term
#     #         self.integral += (pix_error) * dt
#     #         self.linear_integral += (distance_error) * dt
            
#     #         # derivative term
#     #         derivative = (pix_error - self.last_error) / dt
#     #         linear_derivative = (distance_error - self.last_linear_error) / dt

#     #         # pid
#     #         output = - (self.Kp * pix_error + self.Ki * self.integral + self.Kd * derivative)
#     #         control = self.linear_Kp * distance_error + self.linear_Ki * self.linear_integral + self.linear_Kd * linear_derivative
            
#     #         # clamp angular vel
#     #         output = max(min(output, self.max_angular_speed), -self.max_angular_speed)

#     #         # clamp linear vel 
#     #         control_output = max(min(control, self.max_linear_speed), -self.max_linear_speed)
            
#     #         twist.angular.z = output

#     #         twist.linear.x = control_output

#     #         # updates
#     #         self.last_error = pix_error
#     #         self.last_linear_error = distance_error
#     #         self.last_update_time = current_time

#     #     else:
#     #         # stop if in deadband
#     #         twist.angular.z = 0.0
#     #         twist.linear.x = 0.0
        
#     #     self._vel_publish.publish(twist)
        
#     def check_timeout(self): 
#         """Stop movement if no new message is received for timeout_duration seconds."""
#         time_since_last_msg = (self.get_clock().now() - self.last_msg_time).nanoseconds / 1e7  # Convert to seconds
#         if time_since_last_msg > self.timeout_duration:
#             twist = Twist()
#             twist.angular.z = 0.0 
#             twist.linear.x = 0.0  
#             self._vel_publish.publish(twist)

# def main():
#     rclpy.init()
#     velocity_publisher = VelocityController()

#     while rclpy.ok():
#         rclpy.spin_once(velocity_publisher)
    
#     velocity_publisher.destroy_node()  
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()