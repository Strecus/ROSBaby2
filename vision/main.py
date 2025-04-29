import os
import platform
import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
from collections import deque

import hand_gesture
import body_pose
import detector

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# Define the landmarks we're interested in for airplane marshalling
LANDMARKS_OF_INTEREST = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]



def main():
    # Print welcome message
    print("\n=== Gesture Recognition System ===")
    print("This system can recognize gestures using hand tracking or body pose detection.")
    
    # Set paths based on operating system
    system = platform.system()
    model_dir = 'models'

    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Operating system detected: {system}")
    print(f"Using model directory: {model_dir}")
    
    # Ask user for camera ID
    camera_id = 1
    try:
        camera_input = input("\nEnter camera ID (default is 0, use 1 for external camera): ")
        if camera_input.strip():
            camera_id = int(camera_input)
    except ValueError:
        print("Invalid input. Using default camera (ID: 0)")
        camera_id = 0
    
    # Ask user which mode they want to use
    valid_choice = False
    while not valid_choice:
        print("\nChoose recognition mode:")
        print("1. Hand Gesture Recognition")
        print("2. ML-based Body Pose Recognition")
        print("3. Angle-based Body Pose Recognition")
        choice = input("Enter your choice (1, 2, or 3): ")
        
        if choice == "1":
            print("\nStarting Hand Gesture Recognition...")
            print("Available gestures:")
            print("- Thumb Up: Forward command")
            print("- Thumb Down: Backward command")
            print("- Open Palm: Stop command")
            print("- Pointing Right: Right turn command")
            print("- Pointing Left: Left turn command")
            print("\nPress 'q' to quit the application")
            hand_gesture.run_hand_gesture(camera_id)
            valid_choice = True
           
        elif choice == "2":
            # Sub-menu for body pose detection
            print("Under maintinenece")
            """
            body_pose_choice = ""
            while body_pose_choice not in ["1", "2"]:
                print("\nBody Pose Recognition Options:")
                print("1. Use ML model (if available)")
                print("2. Use basic pose detection logic")
                body_pose_choice = input("Enter your choice (1 or 2): ")
            
            force_basic_detection = (body_pose_choice == "2")
            
            print("\nStarting ML-based Body Pose Recognition...")
            print("Available poses:")
            print("- Arms Up: Forward command")
            print("- Arms Down: Backward command")
            print("- T-Pose (arms out to sides): Stop command")
            print("- Right Arm Out: Right turn command")
            print("- Left Arm Out: Left turn command")
            print("\nPress 'q' to quit the application")
            body_pose.run_body_pose(camera_id, force_basic_detection)
            valid_choice = True
         """   
        elif choice == "3":
            print("\nStarting Angle-based Body Pose Recognition...")
            print("Available poses:")
            print("- Forward: Cross arms in front")
            print("- Turn Left: Left arm bent, right arm straight")
            print("- Turn Right: Right arm bent, left arm straight")
            print("- Stop: Arms straight up")
            print("\nPress 'q' to quit the application")
            detector.run_pose_detection(camera_id)
            valid_choice = True
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\nApplication closed.")

if __name__ == "__main__":
    main()
