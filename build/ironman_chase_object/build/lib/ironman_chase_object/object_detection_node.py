# Marilyn Braojos 
# Mariam Misabishvili

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point

import cv2
from cv_bridge import CvBridge
import numpy as np

class ObjectDetectionNode(Node):

    def __init__(self):        
        super().__init__('object_detection_node')
        
        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self._video_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback, 
            image_qos_profile)
        self._video_subscriber 
        
        self._point_publish = self.create_publisher(Point, 'detected_pixel', 10)

    def _image_callback(self, CompressedImage):    
        self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
        pixel_coordinates = self.find_object(self._imgBGR)

        if pixel_coordinates is not None: 
            cx, image_center_x = pixel_coordinates
            point_msg = Point()
            point_msg.x = float(cx)
            point_msg.y = float(image_center_x)
            point_msg.z = 0.0

            self._point_publish.publish(point_msg)
            self.get_logger().info(f"Published pixel center {cx} and frame center {image_center_x}")


    def find_object(self, frame):
        lower_color = np.array([40, 75, 75]) # green objects
        upper_color = np.array([80, 255, 255]) # green objects

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bin_mask = cv2.inRange(hsv_frame, lower_color, upper_color)

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 2000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            mask_roi = bin_mask[y:y + h, x:x + w]
            avg_intensity = np.mean(mask_roi) / 255.0

            if avg_intensity < 0.35:
                continue
            
            image_center_x = frame.shape[1] // 2

            return cx, image_center_x

def main():
    rclpy.init()
    video_subscriber = ObjectDetectionNode()

    while rclpy.ok():
        rclpy.spin_once(video_subscriber)
    
    video_subscriber.destroy_node()  
    rclpy.shutdown()

if __name__ == '__main__':
    main()