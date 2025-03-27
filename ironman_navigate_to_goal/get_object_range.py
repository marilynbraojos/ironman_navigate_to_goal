# # Marilyn Braojos 
# # Mariam Misabishvili

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point

import math
import statistics

class GetObjectRangeNode(Node):

    def __init__(self):        
        super().__init__('get_object_range_node')

        self.camera_fov_deg = 62.2 # camera's fov [deg]
        self.image_width = 320 # image width [pixels]
        self.angle_per_pixel = self.camera_fov_deg / self.image_width 

        self.object_x = None
        self.center_img = None

        self.last_update_time = self.get_clock().now()
        self.create_timer(1.0, self._check_timeout) 

        lidar_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self._lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self._distance_callback, 
            lidar_qos_profile)
        self._lidar_subscriber 
        
        self.object_distance_publisher = self.create_publisher(Point, 'detected_distance', 10)

    def _distance_callback(self, scan_msg: LaserScan):
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        ranges = scan_msg.ranges
        num_ranges = len(ranges)

        # Define front-facing angle bounds in radians
        lower_bound_1 = 0.0
        upper_bound_1 = math.radians(31)
        lower_bound_2 = math.radians(329)
        upper_bound_2 = 2 * math.pi  # 360° in radians

        min_distance = float('inf')
        min_index = -1

        for i in range(num_ranges):
            angle_rad = angle_min + i * angle_increment
            angle_rad_wrapped = angle_rad % (2 * math.pi)

            in_front = (lower_bound_1 <= angle_rad_wrapped <= upper_bound_1) or \
                    (lower_bound_2 <= angle_rad_wrapped <= upper_bound_2)

            if in_front and 0.0 < ranges[i] < float('inf'):
                if ranges[i] < min_distance:
                    min_distance = ranges[i]
                    min_index = i

        if min_index == -1:
            self.get_logger().info("No valid obstacle detected in front.")
            return

        # Get the angle of the closest obstacle in front
        angle_rad = angle_min + min_index * angle_increment 

        x_coord = min_distance * math.cos(angle_rad)
        y_coord = min_distance * math.sin(angle_rad)

        # Convert polar to Cartesian
        point = Point()
        point.x = x_coord # min dis [m]
        point.y = y_coord # angle [rad]
        point.z = 0.0
        self.object_distance_publisher.publish(point)

        self.get_logger().info(f"Closest front obstacle: distance = {min_distance:.2f} m, angle = {angle_rad:.2f} rad. X: {x_coord}, Y:{y_coord}")

    def _check_timeout(self):
        if (self.get_clock().now() - self.last_update_time).nanoseconds > 1:  # [ns]
            if self.object_x is not None:
                self.get_logger().info("No new object location detected — resetting.")
                self.object_x = None

def main():
    rclpy.init()
    lidar_subscriber = GetObjectRangeNode()

    while rclpy.ok():
        rclpy.spin_once(lidar_subscriber)
    
    lidar_subscriber.destroy_node()  
    rclpy.shutdown()

if __name__ == '__main__':
    main()