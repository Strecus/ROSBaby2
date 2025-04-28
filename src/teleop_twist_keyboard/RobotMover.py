#!/usr/bin/env python3


import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

def main():
    rospy.init_node('object_detection_mover', anonymous=True)

    # Publisher for robot velocity
    velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Subscriber for laser scan data
    scan_sub = rospy.Subscriber('/scan', LaserScan, laser_callback)

    # Initialize the Twist message
    move_cmd = Twist()

    rospy.spin()
def laser_callback( msg):
    # Detect if the robot is within a certain range of an object
    # For simplicity, we'll check the front of the robot (center of the laser scan)
    threshold = 1.0  # meters
    if any(r < threshold for r in msg.ranges if r > 0.01):
        move_robot()
        print("Obstacle detected within 1 meter")
    

    # Define a threshold for detecting objects
    object_threshold = 1.0  # in meters

    if front_distance < object_threshold:
        #move_robot()
        print("Detected!")
    else:
        print("Not seen")
        #stop_robot()

def move_robot():
    # Move forward at a constant speed
    move_cmd.linear.x = 0.5  # Move forward with 0.5 m/s speed
    move_cmd.angular.z = 0.0  # No rotation
    velocity_pub.publish(move_cmd)

def stop_robot():
    # Stop the robot
    move_cmd.linear.x = 0.0
    move_cmd.angular.z = 0.0
    velocity_pub.publish(move_cmd)

def run():
    rospy.spin()

if __name__ == '__main__':
    main()
