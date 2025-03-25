# Bare Bones Code to View the Image Published from the Turtlebot3 on a Remote Computer
# Intro to Robotics Research 7785
# Georgia Institute of Technology
# Sean Wilson, 2022

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys

import numpy as np
import cv2
from cv_bridge import CvBridge

class MinimalVideoSubscriber(Node):

	def __init__(self):		
		# Creates the node.
		super().__init__('minimal_video_subscriber')

		# Set Parameters
		self.declare_parameter('show_image_bool', True)
		self.declare_parameter('window_name', "Raw Image")

		#Determine Window Showing Based on Input
		self._display_image = bool(self.get_parameter('show_image_bool').value)

		# Declare some variables
		self._titleOriginal = self.get_parameter('window_name').value # Image Window Title	
		
		#Only create image frames if we are not running headless (_display_image sets this)
		if(self._display_image):
		# Set Up Image Viewing
			cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE ) # Viewing Window
			cv2.moveWindow(self._titleOriginal, 50, 50) # Viewing Window Original Location
		
		#Set up QoS Profiles for passing images over WiFi
		image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

		#Declare that the minimal_video_subscriber node is subcribing to the /camera/image/compressed topic.
		self._video_subscriber = self.create_subscription(
				CompressedImage,
				'/image_raw/compressed',
				self._image_callback,
				image_qos_profile)
		self._video_subscriber # Prevents unused variable warning.

	def _image_callback(self, CompressedImage):	
		# The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
		self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
		if(self._display_image):
			# Display the image in a window
			# self.show_image(self._imgBGR)
			self.find_object(self._imgBGR) # may not work 

	def find_object(self, frame):
		# Color range for detecting green objects
		lower_color = np.array([40, 75, 75])
		upper_color = np.array([80, 255, 255])

		# Convert frame captured to HSV color space
		hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# Check if elements are within the specified color range and create binary mask
		bin_mask = cv2.inRange(hsv_frame, lower_color, upper_color)

		# Find contours
		contours, _ = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# For all contours found
		for contour in contours:
			area = cv2.contourArea(contour)
			if area < 2000:  # filter area of objects to detect
				continue

			x, y, w, h = cv2.boundingRect(contour)

			mask_roi = bin_mask[y:y + h, x:x + w]
			avg_intensity = np.mean(mask_roi) / 255.0

			# If the contour's average intensity is too dark, skip it
			if avg_intensity < 0.35:
				continue

			# Draw bounding box and label
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, f"Object Detected at Position: ({x}, {y})", (x, y - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		# Display the processed frame with detections
		if self._display_image:
			cv2.imshow(self._titleOriginal, frame)	
			self._user_input=cv2.waitKey(50) #Use OpenCV keystroke grabber for delay.	

	# def show_image(self, img):
	# 	cv2.imshow(self._titleOriginal, img)
	# 	# Cause a slight delay so image is displayed
	# 	self._user_input=cv2.waitKey(50) #Use OpenCV keystroke grabber for delay.

	def get_user_input(self):
		return self._user_input

def main():
	rclpy.init() #init routine needed for ROS2.
	video_subscriber = MinimalVideoSubscriber() #Create class object to be used.

	while rclpy.ok():
		rclpy.spin_once(video_subscriber) # Trigger callback processing.
		if(video_subscriber._display_image):	
			if video_subscriber.get_user_input() == ord('q'):
				cv2.destroyAllWindows()
				break
	rclpy.logging.get_logger("Camera Viewer Node Info...").info("Shutting Down")
	#Clean up and shutdown.
	video_subscriber.destroy_node()  
	rclpy.shutdown()


if __name__ == '__main__':
	main()
