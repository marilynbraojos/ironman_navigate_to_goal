# Marilyn Braojos 
# Mariam Misabishvili

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
        self.create_timer(1.0, self._check_timeout)  # Check every second

        lidar_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self._pix_subscriber = self.create_subscription(
            Point,
            'detected_pixel',
            self._pixel_callback, 
            10)
        self._pix_subscriber 

        self._lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self._distance_callback, 
            lidar_qos_profile)
        self._lidar_subscriber 
        
        self.object_distance_publisher = self.create_publisher(Point, 'detected_distance', 10)

    def _pixel_callback(self, msg: Point):
        self.object_x = msg.x # object center in pixels
        self.center_img = msg.y # img center in pixels
        self.last_update_time = self.get_clock().now()

    def _distance_callback(self, scan_msg: LaserScan):    
        if self.object_x is None: 
            return 

        pix_error = self.center_img - self.object_x
        angle_deg = pix_error * self.angle_per_pixel
        angle_rad = math.radians(angle_deg)

        if pix_error < 0:
            angle_rad = math.radians(angle_deg) + (2 * math.pi)

        # Find the indices within +/- 0.2 rad
        min_angle = angle_rad - 0.1
        max_angle = angle_rad + 0.1

        indices = [i for i in range(len(scan_msg.ranges)) 
                   if scan_msg.angle_min + i * scan_msg.angle_increment >= min_angle 
                   and scan_msg.angle_min + i * scan_msg.angle_increment <= max_angle]
                
        # Get the average distance
        valid_ranges = [scan_msg.ranges[i] for i in indices if scan_msg.ranges[i] > 0.0]
        
        if valid_ranges:
            distance = statistics.median(valid_ranges)
            point = Point()
            point.x = -pix_error
            point.y = distance
            self.object_distance_publisher.publish(point)      

        # Debugging
        self.get_logger().info(f"pix_error: {pix_error}")
        self.get_logger().info(f"angle_deg: {angle_deg}")
        self.get_logger().info(f"angle_rad: {angle_rad}")
        self.get_logger().info(f"Indices in window: {indices}")
        self.get_logger().info(f"Raw distances: {valid_ranges} m")
        self.get_logger().info(f"Median distance: {distance} m")
        # self.get_logger().info(f"Min Range: {scan_msg.range_min} m")
        # self.get_logger().info(f"Max Range: {scan_msg.range_max} m")
        # self.get_logger().info(f"Ranges: {scan_msg.ranges} m")


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