# Marilyn Braojos 
# Mariam Misabishvili

import rclpy
from rclpy.node import Node  
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist, Vector3Stamped 
import math
import numpy as np

# WAYPOINTS_FILE = "wayPoints.txt" # file containing waypoints
# WAYPOINT_TOLERANCES = [0.05, 0.05, 0.05, 0.05] # waypoint tolerances
# STOP_DURATIONS = [10, 10, 2, 10] # stopping time at each waypoint

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')  # initialize node

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)  # create published for robot vel
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10) # subscribe to odometry readings
        self.obs_sub = self.create_subscription(Vector3Stamped, '/detected_distance', self.lidar_callback, 10) # subscribe to object location

        # self.waypoints = self.read_waypoints() # load waypoints 

        # self.goal_index = 0 # index of current waypoint
        self.yaw = 0.0 # initialize orientation
        self.init_yaw = 0.0 # orientation from the start
        self.init_x = 0.0 # initialize starting position
        self.init_y = 0.0 # initialize starting position
        self.avoid_step = 0 # step in the avoidance logic
        self.current_position = (0.0, 0.0) # initialize transformed robot position
        
        self.odom_offset = None # flag to trigger offset at start
        self.init_pos = None # initialize starting position
        self.rotation_matrix = None # initialize rotate global frame to initial orientation 
        # self.reached_time = None # initialize time goal was reached
        self.obstacle_distance = None # initialize distane of obstacle
        self.avoid_start_time = None # initialize time that avoidance began

        self.avoiding_obstacle = False # initialize whether avoidance is ongoing
        
        # self.start_time = self.get_clock().now().seconds_nanoseconds()[0] # global timer
        self.timer = self.create_timer(0.1, self.controller_loop) # timer to call the control loop 

    # def read_waypoints(self): # read waypoints from text file
    #     waypoints = [] # initialize list
    #     with open(WAYPOINTS_FILE, 'r') as f: # read file 
    #         for line in f: # for every line in the file
    #             x, y = map(float, line.strip().split()) # x and y are separated by a space
    #             waypoints.append((x, y)) # add the read x and y to the list
    #     return waypoints # return the waypoints when this fcn is called

    def odom_callback(self, msg): # setting up initial odometry offset for frame transforms 
        position = msg.pose.pose.position # get current position 
        q = msg.pose.pose.orientation # get current orientation (quaternions)
        orientation = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)) # convert quaternion to yaw 

        if self.odom_offset is None: # if we havent initialized the starting position and orientation
            self.odom_offset = True # set initialization to True
            self.init_yaw = orientation # store the yaw angle from the start
            self.init_pos = position # store the starting position 
            self.rotation_matrix = np.array([ 
                [np.cos(self.init_yaw), np.sin(self.init_yaw)],
                [-np.sin(self.init_yaw), np.cos(self.init_yaw)]
            ]) # transform global odom to robot frame

            # apply rotation matrix 
            self.init_x = self.rotation_matrix[0, 0] * position.x + self.rotation_matrix[0, 1] * position.y
            self.init_y = self.rotation_matrix[1, 0] * position.x + self.rotation_matrix[1, 1] * position.y

        # apply rotation matrix to current position 
        transformed_x = self.rotation_matrix[0, 0] * position.x + self.rotation_matrix[0, 1] * position.y
        transformed_y = self.rotation_matrix[1, 0] * position.x + self.rotation_matrix[1, 1] * position.y

        # subtract the transformed initial position to get current position relative to the start
        self.current_position = (transformed_x - self.init_x, transformed_y - self.init_y)
        self.yaw = orientation - self.init_yaw

    def lidar_callback(self, msg: Vector3Stamped):
        self.obstacle_distance = msg.vector.z # obtain obstacle distance
        if self.obstacle_distance < 0.20 and not self.avoiding_obstacle: # if the obstacle's distance is less than 20 cm and avoidance isn't ongoing currently 
            self.avoiding_obstacle = True # initiate obstacle avoidance
            self.avoid_step = 0 # flag first step in the logic 

    def controller_loop(self): # main control loop running every 0.1 s
        now = self.get_clock().now().nanoseconds / 1e9 # obtaining current time 

        # Timeout check
        # if self.goal_index >= len(self.waypoints) or (now - self.start_time > 150): # if the goal index exceeds the len of the waypoints or if 2 min 30 s reached, terminate
            # self.cmd_pub.publish(Twist()) # stop robot motors
            # return

        # Obstacle avoidance logic
        if self.avoiding_obstacle: # if obstacle avoidance is initiated
            if self.avoid_step == 0: # if at step 1 of the process
                self.get_logger().info("Obstacle detected. Backing up") # warn used avoidance is starting
                self.avoid_step = 1 # go to step 2

            elif self.avoid_step == 1: # if at step 2
                if self.obstacle_distance is not None and self.obstacle_distance < 0.40: # if avoidance is in progress and the object is less than 40 cm away
                    cmd = Twist()
                    cmd.linear.x = -0.1 # back up 
                    self.cmd_pub.publish(cmd)
                    return # continue this process until object is above 40 cm away 
                else:
                    self.get_logger().info("🔙 Backup complete. Preparing to left.") # inform user back up is completed
                    self.avoid_step = 2 # go to step 3 
                    self.avoid_start_time = now # initialize timer 
                    return

            elif self.avoid_step == 2: # if at step 3 
                if self.avoid_start_time is None: # if the timer didnt initiate 
                    self.avoid_start_time = now # start timer here

                if now - self.avoid_start_time < 2.0: # if the step has been going on for less than 2 seconds 
                    cmd = Twist() 
                    cmd.angular.z = 0.5 # turn left until 2 seconds are reached
                    self.cmd_pub.publish(cmd)
                    self.get_logger().info("↩️ Turning left") # inform user 
                    return
                elif now - self.avoid_start_time < 5.0: # if the time is less than 5 seconds but more than 2 seconds
                    cmd = Twist()
                    cmd.linear.x = 0.15 # move forward 
                    self.cmd_pub.publish(cmd)
                    self.get_logger().info("Moving forward")
                    return
                else:
                    self.get_logger().info("✅ Avoidance maneuver complete. Resuming navigation.") # after 5 seconds - complete avoidance maneuver
                    self.avoiding_obstacle = False # reset obstacle detected flag
                    self.avoid_step = 0 # reset avoidance step 
                    self.avoid_start_time = None # reset timer 
                    self.cmd_pub.publish(Twist()) # stop robot 
                    return
        # else: 
        #     cmd = Twist()
        #     cmd.linear.x = 0.15
        #     self.cmd_pub.publish(cmd)


        # Waypoint navigation
        # goal = self.waypoints[self.goal_index] # get current waypoint
        # dx = goal[0] - self.current_position[0] # difference in x from current to goal 
        # dy = goal[1] - self.current_position[1] # difference in y from current to goal 
        # distance = math.sqrt(dx**2 + dy**2) # evaluate distance to waypoint

        # if distance < WAYPOINT_TOLERANCES[self.goal_index]: # if the distance is not within tolerance
            # if self.reached_time is None: 
                # self.reached_time = now # starting the waiting at each position timer 
            # elif now - self.reached_time >= STOP_DURATIONS[self.goal_index]: # move on to next index after timer is done
                # self.goal_index += 1
                # self.reached_time = None # reset timer 
            # self.cmd_pub.publish(Twist())
            # return

        # angle_to_goal = math.atan2(dy, dx) # angle from robot to goal 
        # angle_diff = math.atan2(math.sin(angle_to_goal - self.yaw), math.cos(angle_to_goal - self.yaw)) # get shortest rotation direction to the goal

        cmd = Twist()
        cmd.linear.x = 0.15
        # if abs(angle_diff) > 0.1:
            # cmd.angular.z = 0.5 * angle_diff
        # else:
            # cmd.linear.x = 0.15
            # cmd.angular.z = 0.3 * angle_diff

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = GoToGoal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()