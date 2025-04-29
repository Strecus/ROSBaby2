import cv2
import mediapipe as mp
import numpy as np
import time
import os
import platform
import sys
sys.path.append('/home/strecus/jackal_ws/src/the-barn-challenge/')
import cv_commands_publisher


# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Map gestures to commands
GESTURE_MAP = {
    "Thumb_Up": "forward",
    "Thumb_Down": "backward",
    "Open_Palm": "stop",
    "Pointing_Right": "right",
    "Pointing_Left": "left"
}
def sendGoal(gesture_type):
    """
    Call the goal publisher or execute path from cv_commands_publisher
    with the given gesture type, based on user input.
    """
    if gesture_type.lower() == "no pose detected" or gesture_type.lower() == "unknown":
        return
    try:
        #choice = input("Do you want to control the robot directly instead of sending goals? (y/n): ").strip().lower()
        print(gesture_type)
        cv_commands_publisher.publish_goal(gesture_type)
        """
        choice = 'n'
        if choice == 'y':
            cv_commands_publisher.execute_path(gesture_type)
        else:
            cv_commands_publisher.publish_goal(gesture_type)
        """
    except Exception as e:
        print(f"Error handling goal command: {e}")

# Helper function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

# Helper function to calculate angle between three points
def calculate_angle(point1, point2, point3):
    # Create vectors
    v1 = np.array([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])
    v2 = np.array([point3.x - point2.x, point3.y - point2.y, point3.z - point2.z])
    
    # Calculate the angle using the dot product formula
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)

# Helper function to normalize confidence value to a 0-1 range
def normalize_confidence(value, min_val, max_val):
    return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)

def run_hand_gesture(camera_id=0):
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
    print("Starting hand gesture recognition mode...")
    
    # Start webcam
    cap = cv2.VideoCapture(camera_id)  # Use specified camera ID
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return False

    # Configure MediaPipe Hands with appropriate confidence thresholds
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        # For FPS calculation
        prev_time = 0
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # To improve performance, optionally mark the image as not writeable
            rgb_frame.flags.writeable = False
            
            # Process the image and detect hands
            results = hands.process(rgb_frame)
            
            # Draw the hand annotations on the image
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Extract gesture and display on frame
            gesture_detected = "No gesture detected"
            confidence = 0.0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get MediaPipe detection confidence from the result
                    if results.multi_handedness:
                        for hand_idx, hand_handedness in enumerate(results.multi_handedness):
                            # The score is the confidence of handedness detection
                            # We'll use this as a base confidence factor
                            base_confidence = hand_handedness.classification[0].score
                        
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract key landmarks
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    
                    # Get finger MCP joints (knuckles)
                    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                    
                    # Get PIP joints (middle knuckles)
                    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
                    
                    # Calculate distances for confidence metrics
                    thumb_wrist_dist = calculate_distance(thumb_tip, wrist)
                    index_mcp_dist = calculate_distance(index_tip, index_mcp)
                    middle_mcp_dist = calculate_distance(middle_tip, middle_mcp)
                    ring_mcp_dist = calculate_distance(ring_tip, ring_mcp)
                    pinky_mcp_dist = calculate_distance(pinky_tip, pinky_mcp)
                    
                    # Thumb up detection - Forward command
                    if (thumb_tip.y < wrist.y and 
                        index_tip.y > index_mcp.y and 
                        middle_tip.y > middle_mcp.y and 
                        ring_tip.y > ring_mcp.y and 
                        pinky_tip.y > pinky_mcp.y):
                        gesture_detected = "forward"  # Thumb up
                        
                        # Calculate confidence based on how extended the thumb is and how curled other fingers are
                        thumb_extension = normalize_confidence(wrist.y - thumb_tip.y, 0.05, 0.2)
                        finger_curl = normalize_confidence(
                            np.mean([index_tip.y - index_mcp.y, 
                                    middle_tip.y - middle_mcp.y,
                                    ring_tip.y - ring_mcp.y,
                                    pinky_tip.y - pinky_mcp.y]), 0.01, 0.1)
                        
                        # Combine factors for overall confidence
                        confidence = base_confidence * 0.3 + thumb_extension * 0.4 + finger_curl * 0.3
                        confidence = confidence * 100  # Convert to percentage
                    
                    # Thumb down detection - Backward command
                    elif (thumb_tip.y > wrist.y and 
                          index_tip.y > index_mcp.y and 
                          middle_tip.y > middle_mcp.y and 
                          ring_tip.y > ring_mcp.y and 
                          pinky_tip.y > pinky_mcp.y):
                        gesture_detected = "backward"  # Thumb down
                        
                        # Calculate confidence based on how downward the thumb is and how curled other fingers are
                        thumb_extension = normalize_confidence(thumb_tip.y - wrist.y, 0.05, 0.2)
                        finger_curl = normalize_confidence(
                            np.mean([index_tip.y - index_mcp.y, 
                                    middle_tip.y - middle_mcp.y,
                                    ring_tip.y - ring_mcp.y,
                                    pinky_tip.y - pinky_mcp.y]), 0.01, 0.1)
                        
                        # Combine factors for overall confidence
                        confidence = base_confidence * 0.3 + thumb_extension * 0.4 + finger_curl * 0.3
                        confidence = confidence * 100  # Convert to percentage
                    
                    # Open palm detection - Stop command
                    elif (thumb_tip.y < thumb_mcp.y and 
                          index_tip.y < index_mcp.y and 
                          middle_tip.y < middle_mcp.y and 
                          ring_tip.y < ring_mcp.y and 
                          pinky_tip.y < pinky_mcp.y):
                        gesture_detected = "stop"  # Open palm
                        
                        # Calculate confidence based on how extended all fingers are
                        finger_extensions = [
                            normalize_confidence(thumb_mcp.y - thumb_tip.y, 0.02, 0.15),
                            normalize_confidence(index_mcp.y - index_tip.y, 0.05, 0.2),
                            normalize_confidence(middle_mcp.y - middle_tip.y, 0.05, 0.2),
                            normalize_confidence(ring_mcp.y - ring_tip.y, 0.05, 0.2),
                            normalize_confidence(pinky_mcp.y - pinky_tip.y, 0.05, 0.2)
                        ]
                        
                        # Combine factors for overall confidence
                        extension_confidence = np.mean(finger_extensions)
                        
                        # Check finger spread as well
                        finger_spread = normalize_confidence(
                            calculate_distance(index_tip, pinky_tip), 0.1, 0.3)
                        
                        confidence = base_confidence * 0.3 + extension_confidence * 0.4 + finger_spread * 0.3
                        confidence = confidence * 100  # Convert to percentage
                    
                    # Pointing right - Right command
                    elif (index_tip.x > index_mcp.x and  # Index finger is to the right of its base
                          (index_tip.x - index_mcp.x) > 0.5 * abs(index_tip.y - index_mcp.y)):  # Less strict horizontal requirement
                        gesture_detected = "right"  # Pointing right
                        
                        # Calculate confidence based on how extended the index finger is horizontally
                        index_horizontal_extension = normalize_confidence(index_tip.x - index_mcp.x, 0.03, 0.15)
                        horizontal_ratio = (index_tip.x - index_mcp.x) / max(0.001, abs(index_tip.y - index_mcp.y))
                        horizontal_confidence = normalize_confidence(horizontal_ratio, 0.8, 2.0)
                        
                        # Combine factors for overall confidence
                        confidence = (base_confidence * 0.3 + 
                                     index_horizontal_extension * 0.4 + 
                                     horizontal_confidence * 0.3)
                        confidence = confidence * 100  # Convert to percentage
                    
                    # Pointing left - Left command
                    elif (index_tip.x < index_mcp.x and  # Index finger is to the left of its base
                          (index_mcp.x - index_tip.x) > 0.5 * abs(index_tip.y - index_mcp.y)):  # Less strict horizontal requirement
                        gesture_detected = "left"  # Pointing left
                        
                        # Calculate confidence based on how extended the index finger is horizontally
                        index_horizontal_extension = normalize_confidence(index_mcp.x - index_tip.x, 0.03, 0.15)
                        horizontal_ratio = (index_mcp.x - index_tip.x) / max(0.001, abs(index_tip.y - index_mcp.y))
                        horizontal_confidence = normalize_confidence(horizontal_ratio, 0.8, 2.0)
                        
                        # Combine factors for overall confidence
                        confidence = (base_confidence * 0.3 + 
                                     index_horizontal_extension * 0.4 + 
                                     horizontal_confidence * 0.3)
                        confidence = confidence * 100  # Convert to percentage
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Display the gesture name on the frame
            cv2.putText(frame, f"Command: {gesture_detected}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display confidence if available
            if confidence > 0:
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the mode
            cv2.putText(frame, "Mode: Hand Gesture", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Gesture Recognition', frame)

            if gesture_detected in GESTURE_MAP.values():
                sendGoal(gesture_detected)
                while True:
                    if cv2.waitKey(1) & 0xFF == ord('n'):
                        cap.release()
                        cap = cv2.VideoCapture(camera_id)
                        break
                        
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    run_hand_gesture() 
