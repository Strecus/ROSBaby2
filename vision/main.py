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

import sys
sys.path.append('/home/strecus/jackal_ws/src/the-barn-challenge/')
import cv_commands_publisher

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


# Utility to calculate angle at point b
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Detect TURN LEFT or TURN RIGHT
def detect_turn_pose(pose_landmarks, mp_pose, side="left"):
    if side == "left":
        static_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
        static_elbow = mp_pose.PoseLandmark.RIGHT_ELBOW
        static_wrist = mp_pose.PoseLandmark.RIGHT_WRIST
        moving_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
        moving_elbow = mp_pose.PoseLandmark.LEFT_ELBOW
        moving_wrist = mp_pose.PoseLandmark.LEFT_WRIST
    else:
        static_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
        static_elbow = mp_pose.PoseLandmark.LEFT_ELBOW
        static_wrist = mp_pose.PoseLandmark.LEFT_WRIST
        moving_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
        moving_elbow = mp_pose.PoseLandmark.RIGHT_ELBOW
        moving_wrist = mp_pose.PoseLandmark.RIGHT_WRIST
    # Coordinates
    static_shoulder_coord = [pose_landmarks[static_shoulder].x, pose_landmarks[static_shoulder].y]
    static_elbow_coord = [pose_landmarks[static_elbow].x, pose_landmarks[static_elbow].y]
    static_wrist_coord = [pose_landmarks[static_wrist].x, pose_landmarks[static_wrist].y]
    moving_shoulder_coord = [pose_landmarks[moving_shoulder].x, pose_landmarks[moving_shoulder].y]
    moving_elbow_coord = [pose_landmarks[moving_elbow].x, pose_landmarks[moving_elbow].y]
    moving_wrist_coord = [pose_landmarks[moving_wrist].x, pose_landmarks[moving_wrist].y]
    hip = mp_pose.PoseLandmark.LEFT_HIP if side == "left" else mp_pose.PoseLandmark.RIGHT_HIP
    hip_coord = [pose_landmarks[hip].x, pose_landmarks[hip].y]
    # Angles
    static_shoulder_angle = calculate_angle(hip_coord, static_shoulder_coord, static_wrist_coord)
    static_elbow_angle = calculate_angle(static_shoulder_coord, static_elbow_coord, static_wrist_coord)
    moving_shoulder_angle = calculate_angle(hip_coord, moving_shoulder_coord, moving_wrist_coord)
    moving_elbow_angle = calculate_angle(moving_shoulder_coord, moving_elbow_coord, moving_wrist_coord)
    # Thresholds
    SHOULDER_MIN, SHOULDER_MAX = 60, 200
    STATIC_ELBOW_MIN, STATIC_ELBOW_MAX = 160, 190
    MOVING_ELBOW_MIN, MOVING_ELBOW_MAX = 30, 150
    static_ok = (SHOULDER_MIN <= static_shoulder_angle <= SHOULDER_MAX and
                 STATIC_ELBOW_MIN <= static_elbow_angle <= STATIC_ELBOW_MAX)
    moving_ok = (SHOULDER_MIN <= moving_shoulder_angle <= SHOULDER_MAX and
                 MOVING_ELBOW_MIN <= moving_elbow_angle <= MOVING_ELBOW_MAX)
    if static_ok and moving_ok and abs(static_elbow_angle - moving_elbow_angle) >= 50:
        return True
    return False

# Detect FORWARD pose
def detect_forward_pose(pose_landmarks, mp_pose):
    # Shoulders, elbows, wrists coords
    ls = [pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
          pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    le = [pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
          pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    lw = [pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
          pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
    rs = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
          pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    re = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
          pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    rw = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
          pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    # Angles
    left_shoulder_angle = calculate_angle(
        [pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
        ls, lw)
    right_shoulder_angle = calculate_angle(
        [pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
        rs, rw)
    left_elbow_angle = calculate_angle(ls, le, lw)
    right_elbow_angle = calculate_angle(rs, re, rw)
    # Tight thresholds
    if (80 <= left_shoulder_angle <= 100 and 80 <= right_shoulder_angle <= 100 and
        60 <= left_elbow_angle <= 100 and 60 <= right_elbow_angle <= 100):
        return True
    # Loose thresholds
    if (50 <= left_shoulder_angle <= 150 and 50 <= right_shoulder_angle <= 150 and
        50 <= left_elbow_angle <= 190 and 50 <= right_elbow_angle <= 190 and
        abs(left_elbow_angle - right_elbow_angle) <= 30):
        return True
    return False
#HELPER FOR DEBUGGING
def draw_pose_debug(frame, landmarks, visibility_threshold=0.5):
    def draw_point(name, color=GREEN, radius=5):
        landmark = landmarks[mp_pose.PoseLandmark[name].value]
        if landmark.visibility > visibility_threshold:
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), radius, color, -1)
            cv2.putText(frame, name, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Key points to debug
    for name in ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                 'LEFT_HIP', 'RIGHT_HIP']:
        draw_point(name)

def run_angle_based_pose(camera_id=0):
    """Run angle-based pose detection"""
    POSE_HISTORY = deque(maxlen=10) # Buffer of last 10 poses
    
    # Start webcam
    cap = cv2.VideoCapture(camera_id)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    # Configure MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Error: Failed to capture image")
                break
            
            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # To improve performance, optionally mark the image as not writeable
            rgb_frame.flags.writeable = False
            
            # Process the image and detect poses
            results = pose.process(rgb_frame)
            
            # Draw the pose annotations on the image
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Extract pose and display on frame
            pose_detected = "No pose detected"
            confidence = 0.0
            
            if results.pose_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                #DEBUGGING HELPER
                draw_pose_debug(frame, results.pose_landmarks.landmark)
                
                # Detect the pose using angle-based detection
                curr_pose = detect_pose(results.pose_landmarks.landmark, mp_pose)

                POSE_HISTORY.append(curr_pose)

                if POSE_HISTORY:
                    pose_detected = max(set(POSE_HISTORY), key=POSE_HISTORY.count)
                else:
                    pose_detected = "unknown"
         
            # Display the pose name on the frame
            cv2.putText(frame, f"Pose: {pose_detected}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display confidence if available
            if confidence > 0:
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Angle-Based Pose Recognition', frame)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Print welcome message
    print("\n=== Gesture Recognition System ===")
    print("This system can recognize gestures using hand tracking or body pose detection.")
    
    # Set paths based on operating system
    system = platform.system()
    if system == 'Darwin':  # macOS
        model_dir = 'models'
    else:  # Windows or other OS
        model_dir = 'models'

    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Operating system detected: {system}")
    print(f"Using model directory: {model_dir}")
    
    # Ask user for camera ID
    camera_id = 0
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
            
        elif choice == "3":
            print("\nStarting Angle-based Body Pose Recognition...")
            print("Available poses:")
            print("- Forward: Cross arms in front")
            print("- Turn Left: Left arm bent, right arm straight")
            print("- Turn Right: Right arm bent, left arm straight")
            print("- Stop: Arms straight up")
            print("\nPress 'q' to quit the application")
            run_angle_based_pose(camera_id)
            valid_choice = True
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\nApplication closed.")

if __name__ == "__main__":
    main()
