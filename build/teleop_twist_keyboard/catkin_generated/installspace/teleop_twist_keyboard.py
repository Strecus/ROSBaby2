#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import time
import math
import json

# Key bindings for manual movement
move_bindings = {
    'w': (1, 0, 0),  # Move forward
    's': (-1, 0, 0),  # Move backward
    'a': (0, 1, 0),  # Turn left
    'd': (0, -1, 0),  # Turn right
    ' ': (0, 0, 0)  # Stop
}

# Speed settings
speed = 0.5  # Linear speed in meters per second
turn = 1.0  # Angular speed in radians per second

# Initialize ROS node and publisher
rospy.init_node('teleop_twist_keyboard')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rate = rospy.Rate(10)  # 10 Hz

# Path storage
path = []

def get_key():
    """Get key from terminal"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def save_path(filename="path.json"):
    """Save the recorded path to a file"""
    with open(filename, 'w') as f:
        json.dump(path, f)
    print(f"Path saved to {filename}")

def load_path(filename="path.json"):
    """Load a recorded path from a file"""
    global path
    with open(filename, 'r') as f:
        path = json.load(f)
    print(f"Path loaded from {filename}")

def execute_path():
    """Replay the recorded path"""
    twist = Twist()
    start_time = time.time()
    for cmd in path:
        linear, angular, duration = cmd
        twist.linear.x = linear
        twist.angular.z = angular
        # Replay each command
        end_time = start_time + duration
        while time.time() < end_time:
            pub.publish(twist)
            rate.sleep()
        # Stop after each movement
        twist.linear.x = 0
        twist.angular.z = 0
        pub.publish(twist)

def main():
    global path
    print("Control Your Robot!")
    print("Press 'w', 's', 'a', 'd' to move.")
    print("Press 'SPACE' to stop.")
    print("Press 'p' to follow a predefined path.")
    print("Press 'r' to record current path.")
    print("Press 'l' to load a saved path.")
    
    twist = Twist()
    while not rospy.is_shutdown():
        key = get_key()
        
        if key in move_bindings:
            x, th, _ = move_bindings[key]
            twist.linear.x = x * speed
            twist.angular.z = th * turn
            pub.publish(twist)
            # Log the movement command (linear, angular, duration)
            path.append((x * speed, th * turn, 1))  # Default duration of 1 second for each move
            
        elif key == 'p':
            print("Executing predefined path (arc)...")
            execute_path()
        
        elif key == 'r':
            print("Recording current path...")
            save_path()  # Save the path to a file

        elif key == 'l':
            print("Loading saved path...")
            load_path()  # Load the path from a file
        
        else:
            twist.linear.x = 0
            twist.angular.z = 0
            pub.publish(twist)
        
        if key == '\x03':  # Ctrl+C to quit
            break

if __name__ == '__main__':
    main()
