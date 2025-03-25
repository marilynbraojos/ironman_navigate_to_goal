import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point

class VelocityController(Node):
    def __init__(self):        
        super().__init__('velocity_publisher')

        # Subscribe to the object distance topic.
        self.distance_subscriber = self.create_subscription(
            Point,
            'detected_distance',
            self.pixel_callback, 
            10)
        self.distance_subscriber 
        
        # Publisher for velocity commands.
        self._vel_publish = self.create_publisher(Twist, '/cmd_vel', 10)

        # PID gains for angular control. - done using 
        self.Kp = 0.005    # Proportional gain [tuned]
        self.Ki = 0.001    # Integral gain [tuned]
        self.Kd = 0.001    # Derivative gain [tuned]

        self.max_angular_speed = 1.0 
        self.dead_zone = 20  # Pixels within which we don't rotate

        self.dead_zone_linear = .05  # m within which we don't rotate

        # Linear control parameters. --- tuning now
        self.linear_Kp = 0.5 # Proportional gain for linear velocity
        self.linear_Ki = 0.001
        self.linear_Kd = 0.001
        
        self.target_distance = 0.4  # [m]
        self.max_linear_speed = 0.3 # [should be .1 m/s]

        # Initialize error tracking for angular PID. - done using
        self.last_error = 0.0
        self.integral = 0.0
        self.last_update_time = self.get_clock().now()

        # Initialize error tracking for linear PID. --- tuning now 
        self.last_linear_error = 0.0
        self.linear_integral = 0.0
        self.last_linear_update_time = self.get_clock().now()

        # Timeout mechanism.
        self.last_msg_time = self.get_clock().now()  # Track last received message time
        self.timeout_duration = 1.0  # Time (in seconds) before stopping motion
        self.timer = self.create_timer(0.5, self.check_timeout)

    def pixel_callback(self, msg: Point):
        current_time = self.get_clock().now()
        self.last_msg_time = current_time  # Update last received message time

        pix_error = msg.x

        distance = msg.y
        distance_error = distance - self.target_distance
        
        twist = Twist()

        dt = (current_time-self.last_update_time).nanoseconds / 1e9 # Time difference in seconds
        
        # Only compute PID if the error exceeds the dead zone.
        if (abs(pix_error) > self.dead_zone or abs(distance_error) > self.dead_zone_linear) and dt > 0.0:
            
            # Accumulate the integral term.
            self.integral += (pix_error) * dt
            self.linear_integral += (distance_error) * dt
            
            # Compute the derivative term.
            derivative = (pix_error - self.last_error) / dt
            linear_derivative = (distance_error - self.last_linear_error) / dt

            # Compute the full PID control output.
            output = - (self.Kp * pix_error + self.Ki * self.integral + self.Kd * derivative)
            control = self.linear_Kp * distance_error + self.linear_Ki * self.linear_integral + self.linear_Kd * linear_derivative
            
            # Clamp the angular velocity.
            output = max(min(output, self.max_angular_speed), -self.max_angular_speed)

            # clamp linear vel 
            control_output = max(min(control, self.max_linear_speed), -self.max_linear_speed)
            
            twist.angular.z = output

            twist.linear.x = control_output

            # Update error for next iteration.
            self.last_error = pix_error
            self.last_linear_error = distance_error
            self.last_update_time = current_time

        else:
            # If within dead zone, no rotation.
            twist.angular.z = 0.0
            twist.linear.x = 0.0
        
        self._vel_publish.publish(twist)
        
    def check_timeout(self): 
        """Stop movement if no new message is received for timeout_duration seconds."""
        time_since_last_msg = (self.get_clock().now() - self.last_msg_time).nanoseconds / 1e7  # Convert to seconds
        if time_since_last_msg > self.timeout_duration:
            twist = Twist()
            twist.angular.z = 0.0  # Stop rotation
            twist.linear.x = 0.0  # Stop linear movement
            self._vel_publish.publish(twist)

def main():
    rclpy.init()
    velocity_publisher = VelocityController()

    while rclpy.ok():
        rclpy.spin_once(velocity_publisher)
    
    velocity_publisher.destroy_node()  
    rclpy.shutdown()

if __name__ == '__main__':
    main()