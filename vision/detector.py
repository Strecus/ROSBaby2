import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
import os
from collections import deque

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

# Define the landmarks we're interested in for airplane marshalling (for ex, wrist, elbows, shoulders)
LANDMARKS_OF_INTEREST = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]

# Load trained model and scaler
model_path = 'training/models/pose_classifier.pkl'
scaler_path = 'training/models/scaler.pkl'



def sendGoal(type, mother=False):
    if !mother:
        cv_commands_publisher.publish_goal(type)
    else:
        cv_commands_publisher.execute_path(type)
        
#TESTING ANGLES START 

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
        # Turn left => Right arm is static, left arm is moving
        static_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
        static_elbow = mp_pose.PoseLandmark.RIGHT_ELBOW
        static_wrist = mp_pose.PoseLandmark.RIGHT_WRIST

        moving_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
        moving_elbow = mp_pose.PoseLandmark.LEFT_ELBOW
        moving_wrist = mp_pose.PoseLandmark.LEFT_WRIST
    elif side == "right":
        # Turn right => Left arm is static, right arm is moving
        static_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
        static_elbow = mp_pose.PoseLandmark.LEFT_ELBOW
        static_wrist = mp_pose.PoseLandmark.LEFT_WRIST

        moving_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
        moving_elbow = mp_pose.PoseLandmark.RIGHT_ELBOW
        moving_wrist = mp_pose.PoseLandmark.RIGHT_WRIST
    else:
        raise ValueError("side must be 'left' or 'right'")

    # Get coordinates
    static_shoulder_coord = [pose_landmarks[static_shoulder].x, pose_landmarks[static_shoulder].y]
    static_elbow_coord = [pose_landmarks[static_elbow].x, pose_landmarks[static_elbow].y]
    static_wrist_coord = [pose_landmarks[static_wrist].x, pose_landmarks[static_wrist].y]

    moving_shoulder_coord = [pose_landmarks[moving_shoulder].x, pose_landmarks[moving_shoulder].y]
    moving_elbow_coord = [pose_landmarks[moving_elbow].x, pose_landmarks[moving_elbow].y]
    moving_wrist_coord = [pose_landmarks[moving_wrist].x, pose_landmarks[moving_wrist].y]

    hip = mp_pose.PoseLandmark.LEFT_HIP if side == "left" else mp_pose.PoseLandmark.RIGHT_HIP
    hip_coord = [pose_landmarks[hip].x, pose_landmarks[hip].y]

    # Calculate angles
    static_shoulder_angle = calculate_angle(
        hip_coord,
        static_shoulder_coord,
        static_wrist_coord
    )
    static_elbow_angle = calculate_angle(static_shoulder_coord, static_elbow_coord, static_wrist_coord)

    moving_shoulder_angle = calculate_angle(
        hip_coord,
        moving_shoulder_coord,
        moving_wrist_coord
    )
    moving_elbow_angle = calculate_angle(moving_shoulder_coord, moving_elbow_coord, moving_wrist_coord)

    # Thresholds
    SHOULDER_ANGLE_MIN = 60
    SHOULDER_ANGLE_MAX = 200  #120
    STATIC_ELBOW_STRAIGHT_MIN = 160
    STATIC_ELBOW_STRAIGHT_MAX = 190
    MOVING_ELBOW_BENT_MIN = 30
    MOVING_ELBOW_BENT_MAX = 150  # Allow more range since it's dynamic

    static_arm_ok = (SHOULDER_ANGLE_MIN <= static_shoulder_angle <= SHOULDER_ANGLE_MAX and
                     STATIC_ELBOW_STRAIGHT_MIN <= static_elbow_angle <= STATIC_ELBOW_STRAIGHT_MAX)

    moving_arm_ok = (SHOULDER_ANGLE_MIN <= moving_shoulder_angle <= SHOULDER_ANGLE_MAX and
                     MOVING_ELBOW_BENT_MIN <= moving_elbow_angle <= MOVING_ELBOW_BENT_MAX)
    
    elbow_angle_difference = abs(static_elbow_angle - moving_elbow_angle)
    arms_are_asymmetric = elbow_angle_difference >= 50  # degrees

    if static_arm_ok and moving_arm_ok and arms_are_asymmetric:
        return True

    return False


# Detect FORWARD pose
def detect_forward_pose(pose_landmarks, mp_pose):
    left_shoulder = [pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                     pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    left_elbow = [pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                  pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    left_wrist = [pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                  pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

    right_shoulder = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                      pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    right_elbow = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                   pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    right_wrist = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                   pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

    left_shoulder_angle = calculate_angle(
        [pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
        left_shoulder,
        left_wrist
    )
    right_shoulder_angle = calculate_angle(
        [pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
        right_shoulder,
        right_wrist
    )
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Tight (perfect) thresholds
    SHOULDER_ANGLE_MIN_TIGHT = 80
    SHOULDER_ANGLE_MAX_TIGHT = 100
    ELBOW_BENT_MIN_TIGHT = 60
    ELBOW_BENT_MAX_TIGHT = 100
    ELBOW_STRAIGHT_MIN_TIGHT = 160
    ELBOW_STRAIGHT_MAX_TIGHT = 180

    # Loose (in-motion) thresholds
    SHOULDER_ANGLE_MIN_LOOSE = 50    #60
    SHOULDER_ANGLE_MAX_LOOSE = 150   #140
    ELBOW_MIN_LOOSE = 50
    ELBOW_MAX_LOOSE = 190

    # Perfect Cross Pose (straight or bent)
    bent_arms_forward_tight = (SHOULDER_ANGLE_MIN_TIGHT <= left_shoulder_angle <= SHOULDER_ANGLE_MAX_TIGHT and
                               SHOULDER_ANGLE_MIN_TIGHT <= right_shoulder_angle <= SHOULDER_ANGLE_MAX_TIGHT and
                               ELBOW_BENT_MIN_TIGHT <= left_elbow_angle <= ELBOW_BENT_MAX_TIGHT and
                               ELBOW_BENT_MIN_TIGHT <= right_elbow_angle <= ELBOW_BENT_MAX_TIGHT)

    straight_arms_forward_tight = (SHOULDER_ANGLE_MIN_TIGHT <= left_shoulder_angle <= SHOULDER_ANGLE_MAX_TIGHT and
                                   SHOULDER_ANGLE_MIN_TIGHT <= right_shoulder_angle <= SHOULDER_ANGLE_MAX_TIGHT and
                                   ELBOW_STRAIGHT_MIN_TIGHT <= left_elbow_angle <= ELBOW_STRAIGHT_MAX_TIGHT and
                                   ELBOW_STRAIGHT_MIN_TIGHT <= right_elbow_angle <= ELBOW_STRAIGHT_MAX_TIGHT)

    # Loose motion detection (movement into forward)
    loose_forward_motion = (SHOULDER_ANGLE_MIN_LOOSE <= left_shoulder_angle <= SHOULDER_ANGLE_MAX_LOOSE and
                             SHOULDER_ANGLE_MIN_LOOSE <= right_shoulder_angle <= SHOULDER_ANGLE_MAX_LOOSE and
                             ELBOW_MIN_LOOSE <= left_elbow_angle <= ELBOW_MAX_LOOSE and
                             ELBOW_MIN_LOOSE <= right_elbow_angle <= ELBOW_MAX_LOOSE)
    
    elbow_angle_difference = abs(left_elbow_angle - right_elbow_angle)
    arms_are_symmetric = elbow_angle_difference <= 30  # degrees

    if (bent_arms_forward_tight or straight_arms_forward_tight or loose_forward_motion) and arms_are_symmetric:
        return True

    return False

#Detect STOP pose (arms straight up)
def detect_stop(pose_landmarks, mp_pose):
    left_shoulder = [pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                     pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    left_elbow = [pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                  pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    left_wrist = [pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                  pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

    right_shoulder = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                      pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    right_elbow = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                   pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    right_wrist = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                   pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    
    left_shoulder_angle = calculate_angle(
        [pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
        left_shoulder,
        left_wrist
    )
    right_shoulder_angle = calculate_angle(
        [pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
        right_shoulder,
        right_wrist
    )
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    SHOULDER_ANGLE_MIN = 160    #60
    SHOULDER_ANGLE_MAX = 180   #140
    ELBOW_MIN = 160
    ELBOW_MAX = 180

    arms_straight_up = (SHOULDER_ANGLE_MIN <= left_shoulder_angle <= SHOULDER_ANGLE_MAX and
                             SHOULDER_ANGLE_MIN <= right_shoulder_angle <= SHOULDER_ANGLE_MAX and
                             ELBOW_MIN <= left_elbow_angle <= ELBOW_MAX and
                             ELBOW_MIN <= right_elbow_angle <= ELBOW_MAX)
    if arms_straight_up:
        return True

    return False



# MASTER pose detection function
def detect_pose(pose_landmarks, mp_pose):
    if detect_turn_pose(pose_landmarks, mp_pose, side="left"):
        return "tleft"
    elif detect_turn_pose(pose_landmarks, mp_pose, side="right"):
        return "right"
    elif detect_forward_pose(pose_landmarks, mp_pose):
        return "forward"
    elif detect_stop(pose_landmarks, mp_pose):
        return "stop"
    else:
        return "unknown"



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



POSE_HISTORY = deque(maxlen=10) #Buffer of last 10 poses





def run_pose_detection(camera_id):
    # Start webcam
    cap = cv2.VideoCapture(camera_id)  # Use 0 for default camera, or 0 if that's your webcam

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Configure MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to capture image")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            pose_detected = "No pose detected"
            confidence = 0.0

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                draw_pose_debug(frame, results.pose_landmarks.landmark)

                curr_pose = detect_pose(results.pose_landmarks.landmark, mp_pose)
                POSE_HISTORY.append(curr_pose)

                if POSE_HISTORY:
                    pose_detected = max(set(POSE_HISTORY), key=POSE_HISTORY.count)
                else:
                    pose_detected = "unknown"

            cv2.putText(frame, f"Pose: {pose_detected}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if confidence > 0:
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Pose Recognition', frame)
             
            if pose_detected.lower() != "unknown":
                sendGoal(pose_detected)
                while True:
                        if cv2.waitKey(1) & 0xFF == ord('n'):
                            cap.release()
                            cap = cv2.VideoCapture(0)
                            break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
