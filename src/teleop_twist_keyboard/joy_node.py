#!/usr/bin/env python3
import subprocess
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import time
import json
import threading

# Speed settings
speed = 25  # Linear speed in meters per second
turn = 10  # Angular speed in radians per second

# Path storage
path = []
recording = False
recording_timer = None
joy_data = None
input_thread = None
waiting_for_input = False

# Initialize ROS node and publisher
rospy.init_node('teleop_joystick')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rate = rospy.Rate(10)  # 10 Hz

def record_path(event):
    """Timer callback function to record path at consistent intervals"""
    global path, joy_data
    # Only record if we're in recording mode and have joystick data
    if recording and joy_data is not None:
        forward_backward = joy_data.axes[1]  # Left stick vertical axis
        left_right = joy_data.axes[0]        # Left stick horizontal axis
	        
        # Add command with fixed duration of 0.1s (matches our timer frequency)
        path.append((forward_backward * speed, left_right * turn, rospy.Duration(.1).tosec()))

def get_custom_filename():
    """Non-blocking function to get custom filename input"""
    global waiting_for_input
    
    waiting_for_input = True
    custom_name = input("Enter a custom filename (without extension): ")
    filename = f"{custom_name}.json"
    
    print(f"Saving path as: {filename}")
    with open(filename, 'w') as f:
        json.dump(path, f)
    print(f"Path saved to {filename} with {len(path)} commands.")
    
    waiting_for_input = False

def save_path_with_custom_name():
    """Save the path with a custom filename in a non-blocking way"""
    global input_thread, waiting_for_input
    
    # Don't start a new thread if one is already running
    if waiting_for_input:
        print("Already waiting for input. Please check your terminal.")
        return
        
    # Start a thread to handle the input without blocking the main thread
    input_thread = threading.Thread(target=get_custom_filename)
    input_thread.daemon = True  # Make thread exit when main program exits
    input_thread.start()
    print("Please check your terminal to enter a filename.")

def get_path_selection():
    """Non-blocking function to get path selection input"""
    global waiting_for_input
    import glob
    import os
    
    waiting_for_input = True
    
    # Find all path files
    path_files = glob.glob("*.json")
    
    if not path_files:
        print("No saved paths found.")
        waiting_for_input = False
        return
        
    path_files = sorted(path_files, key=lambda f: os.path.getctime(f), reverse=True)
    
    # Print the available paths
    print("\n=== LOAD PATH ===")
    print("Available paths:")
    for i, file in enumerate(path_files):
        print(f"  {i+1}: {file}")
    
    try:
        choice = input("Enter the number of the path to load (or press Enter to cancel): ")
        
        if choice.strip():
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(path_files):
                selected_path = path_files[choice_index]
                print(f"Loading path: {selected_path}")
                load_path(selected_path)
                execute_path()
            else:
                print("Invalid selection.")
        else:
            print("Selection cancelled.")
            
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Error while loading path: {e}")
        
    waiting_for_input = False

def load_and_execute_custom_path():
    """List available paths and prompt user to select one in a non-blocking way"""
    global input_thread, waiting_for_input
    
    # Don't start a new thread if one is already running
    if waiting_for_input:
        print("Already waiting for input. Please check your terminal.")
        return
        
    # Start a thread to handle the input without blocking the main thread
    input_thread = threading.Thread(target=get_path_selection)
    input_thread.daemon = True
    input_thread.start()
    print("Please check your terminal to select a path.")

def start_recording():
    """Start recording path with timer"""
    global recording, path, recording_timer
    print("Recording started...")
    path = []  # Clear previous path
    recording = True
    # Start timer to record at exact 10Hz in simulation time
    recording_timer = rospy.Timer(rospy.Duration(0.1), record_path)

def stop_recording():
    """Stop recording path and shut down timer"""
    global recording, recording_timer
    print("Recording stopped.")
    recording = False
    if recording_timer:
        recording_timer.shutdown()

def callback(data):
    """Callback function to handle joystick input"""
    global joy_data, recording
    
    # Store the latest joystick data globally for use by the timer
    joy_data = data
    
    # Buttons for controlling the recording
    x_button = 0  # PS4 'X' button (button 0)
    o_button = 1   # PS4 'O' button (button 1)
    sq_button = 3   # PS4 'Square' button (button 3)
    options_button = 9    # PS4 Start button
    share_button = 8      # PS4 Options button

    # Add conditional checks for these buttons:
    if data.buttons[share_button] == 1:
        # Prompt for filename and load that path
        load_and_execute_custom_path()
    elif data.buttons[x_button] == 1 and not recording:
        # Start recording when 'X' button is pressed
        start_recording()
    elif data.buttons[o_button] == 1 and recording:
        # Stop recording when 'O' button is pressed
        stop_recording()
    elif data.buttons[sq_button] == 1:
        save_path()
    elif data.buttons[options_button] == 1:
        # Prompt for filename and save current path
        save_path_with_custom_name()
    
    # Move the robot based on joystick input regardless of recording state
    twist = Twist()
    twist.linear.x = data.axes[1] * speed
    twist.angular.z = data.axes[0] * turn
    pub.publish(twist)

def save_path(filename="path.json"):
    """Save the recorded path to a file"""
    global path
    with open(filename, 'w') as f:
        json.dump(path, f)
    print(f"Path saved to {filename} with {len(path)} commands.")
    print(f"Estimated path duration: {len(path) * 0.1:.1f} seconds")

def load_path(filename="path.json"):
    """Load a recorded path from a file"""
    global path
    try:
        with open(filename, 'r') as f:
            path = json.load(f)
        print(f"Path loaded from {filename} with {len(path)} commands.")
        print(f"Estimated path duration: {len(path) * 0.1:.1f} seconds")
    except Exception as e:
        print(f"Error loading path: {e}")
        path = []
    
def execute_path(speed_factor=1.0):
    """
    Replay the recorded path
    
    Args:
        speed_factor (float): Multiplier for execution speed.
                             1.0 = original speed, 2.0 = twice as fast
    """
    global path
    twist = Twist()
    
    if not path:
        print("No path to execute. Record a path first.")
        return
        
    print(f"Executing path with {len(path)} commands...")
    
    try:
        for i, cmd in enumerate(path):
            linear, angular, duration = cmd
            
            # Set movement commands
            twist.linear.x = linear
            twist.angular.z = angular
            
            # Publish the command
            pub.publish(twist)
            
            # Sleep using the recorded duration and speed factor
            sleep_duration = duration / speed_factor
            rospy.sleep(sleep_duration)
            
            # Log progress occasionally
            if i % 20 == 0 or i == len(path) - 1:
                print(f"Executing command {i+1}/{len(path)}...")
                
        # Ensure the robot stops at the end
        twist.linear.x = 0
        twist.angular.z = 0
        pub.publish(twist)
        print("Path execution complete.")
        
    except Exception as e:
        # Safety stop if anything goes wrong
        twist.linear.x = 0
        twist.angular.z = 0
        pub.publish(twist)
        print(f"Error during path execution: {e}")

def visualize_path():
    """Print a simple visualization of the recorded path"""
    global path
    if not path:
        print("No path to visualize.")
        return
        
    print("\nRecorded Path Visualization:")
    print("---------------------------")
    for i, cmd in enumerate(path[:20]):  # Show first 20 commands
        linear, angular, duration = cmd
        direction = "↑" if linear > 0 else "↓" if linear < 0 else "·"
        if angular > 0:
            direction = "↰"  # Left turn
        elif angular < 0:
            direction = "↱"  # Right turn
        
        magnitude = abs(linear) / speed  # Normalize by max speed
        visual = direction * max(1, int(magnitude * 10))
        print(f"Step {i:3d}: {visual} (L: {linear:.1f}, A: {angular:.1f}, D: {duration:.3f}s)")
    
    if len(path) > 20:
        print(f"... and {len(path) - 20} more commands")
    print("---------------------------")

def main():
    subprocess.Popen([
    "gnome-terminal", "--",
    "bash", "-c", "rosrun joy joy_node; exec bash"
    ])
    global path
    print("Control Your Robot using Controller!")
    print("Recording your path based on joystick input.")
    print("Press 'X' button to begin recording.")
    print("Press 'O' button to stop recording.")
    print("Press 'Square' button to load and play the last recording.")
    print("Press 'Options' button to save the last path to a customly named file.")
    print("Press 'Share' button to play a path from a customly named file.")
    print("Press '' button to visualize the path.")
    
    # Subscribe to the joystick topic
    rospy.Subscriber("/joy", Joy, callback)
    
    while not rospy.is_shutdown():
        rate.sleep()
if __name__ == '__main__':
    main()
