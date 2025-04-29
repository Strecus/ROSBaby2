import cv2
import mediapipe as mp
import numpy as np
import time
import os
import platform
import pickle
import tensorflow as tf

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Map pose gestures to commands
POSE_MAP = {
    "arms_up": "forward",
    "arms_down": "backward",
    "arms_t": "stop",
    "right_arm_out": "right",
    "left_arm_out": "left"
}

# Define constants for sequence-based prediction
SEQUENCE_LENGTH = 10  # Same as in original implementation

# Define the landmarks we're interested in for airplane marshalling
LANDMARKS_OF_INTEREST = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]

# Define connections between our landmarks of interest
CONNECTIONS_OF_INTEREST = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER)
]

# Helper function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Helper function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# Helper function to normalize confidence value to a 0-1 range
def normalize_confidence(value, min_val, max_val):
    return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)

# Function to extract pose landmarks in the same format as training data
def extract_pose_features(landmarks):
    """Extract pose features from a single frame, using only landmarks of interest"""
    features = []
    for landmark_type in LANDMARKS_OF_INTEREST:
        lm = landmarks[landmark_type.value]
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(features)

# Draw debugging information on the frame for basic pose detection
def draw_pose_debug_info(frame, landmarks):
    """Draw debugging information to help visualize pose thresholds"""
    h, w, _ = frame.shape
    
    # Extract key landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    # Draw horizontal lines at shoulder level
    ls_x, ls_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
    rs_x, rs_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
    cv2.line(frame, (0, ls_y), (w, ls_y), (0, 0, 255), 1)
    
    # Draw vertical arm angles
    lw_x, lw_y = int(left_wrist.x * w), int(left_wrist.y * h)
    rw_x, rw_y = int(right_wrist.x * w), int(right_wrist.y * h)
    
    # Calculate and display arm angles
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    cv2.putText(frame, f"L Angle: {left_arm_angle:.0f}", (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"R Angle: {right_arm_angle:.0f}", (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Calculate vertical distances (negative means wrist is above shoulder)
    left_vertical = left_wrist.y - left_shoulder.y
    right_vertical = right_wrist.y - right_shoulder.y
    
    # Calculate horizontal distances (positive means wrist is to the right of shoulder)
    left_horizontal = left_wrist.x - left_shoulder.x
    right_horizontal = right_wrist.x - right_shoulder.x
    
    cv2.putText(frame, f"L Vert: {left_vertical:.2f}", (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"R Vert: {right_vertical:.2f}", (10, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, f"L Horiz: {left_horizontal:.2f}", (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"R Horiz: {right_horizontal:.2f}", (10, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Basic pose detection function as fallback
def detect_pose_basic(landmarks, frame=None):
    """Basic pose detection logic as fallback if model is not available"""
    
    # Extract key landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    # Get visibility scores
    left_arm_visibility = min(left_shoulder.visibility, left_elbow.visibility, left_wrist.visibility)
    right_arm_visibility = min(right_shoulder.visibility, right_elbow.visibility, right_wrist.visibility)
    
    # Only process if arms are visible enough
    if left_arm_visibility < 0.5 or right_arm_visibility < 0.5:
        return "unknown", 0.0
    
    # Calculate vertical position of wrists relative to shoulders
    # Negative means wrist is above shoulder
    left_vertical = left_wrist.y - left_shoulder.y
    right_vertical = right_wrist.y - right_shoulder.y
    
    # Calculate horizontal position of wrists relative to shoulders
    # Positive means wrist is to the right of shoulder
    left_horizontal = left_wrist.x - left_shoulder.x
    right_horizontal = right_wrist.x - right_shoulder.x
    
    # Print debugging info to console for stop pose
    left_is_horizontal = abs(left_vertical) < 0.15
    right_is_horizontal = abs(right_vertical) < 0.15
    left_is_extended = left_horizontal < -0.1
    right_is_extended = right_horizontal > 0.1
    
    if frame is not None:
        # Add additional debug info
        h, w, _ = frame.shape
        y_pos = 310
        cv2.putText(frame, f"L horizontal: {left_is_horizontal}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"R horizontal: {right_is_horizontal}", (10, y_pos + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"L extended: {left_is_extended}", (10, y_pos + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"R extended: {right_is_extended}", (10, y_pos + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Check for T-pose (arms out to sides, wrists at shoulder level)
    # Relaxed the condition to make it more reliable
    if (abs(left_vertical) < 0.15 and abs(right_vertical) < 0.15 and
        left_horizontal < -0.1 and right_horizontal > 0.1):
        
        # Calculate how horizontal the arms are
        horiz_confidence = 1.0 - (abs(left_vertical) + abs(right_vertical)) / 0.3
        # Calculate how extended the arms are
        extension_confidence = (abs(left_horizontal) + abs(right_horizontal)) / 0.6
        confidence = 0.7 + (horiz_confidence * 0.15) + (extension_confidence * 0.15)
        return "stop", min(confidence, 0.95)
    
    # Check if both arms are raised up (wrists well above shoulders)
    # A negative vertical value means the wrist is above the shoulder
    if left_vertical < -0.1 and right_vertical < -0.1:
        confidence = 0.7 + (min(-left_vertical, -right_vertical) * 0.5)  # Higher confidence if arms higher
        return "forward", min(confidence, 0.95)
    
    # Check if both arms are down (wrists well below shoulders)
    # A positive vertical value means the wrist is below the shoulder
    if left_vertical > 0.15 and right_vertical > 0.15:
        confidence = 0.7 + (min(left_vertical, right_vertical) * 0.5)  # Higher confidence if arms lower
        return "backward", min(confidence, 0.95)
    
    # Check for right arm out (right arm horizontal, left arm down)
    if abs(right_vertical) < 0.15 and right_horizontal > 0.1 and left_vertical > 0.1:
        confidence = 0.7 + (abs(right_horizontal) * 0.5)  # Higher confidence if arm more extended
        return "right", min(confidence, 0.95)
    
    # Check for left arm out (left arm horizontal, right arm down)
    if abs(left_vertical) < 0.15 and left_horizontal < -0.1 and right_vertical > 0.1:
        confidence = 0.7 + (abs(left_horizontal) * 0.5)  # Higher confidence if arm more extended
        return "left", min(confidence, 0.95)
    
    return "unknown", 0.0

# Function to draw only the landmarks we're interested in
def draw_landmarks_of_interest(image, landmarks):
    """Draw only the landmarks we're interested in and their connections"""
    if not landmarks:
        return
        
    h, w, _ = image.shape
    
    # Draw connections
    for connection in CONNECTIONS_OF_INTEREST:
        start_idx = connection[0].value
        end_idx = connection[1].value
        
        start_point = landmarks.landmark[start_idx]
        end_point = landmarks.landmark[end_idx]
        
        # Convert normalized coordinates to pixel coordinates
        start_x, start_y = int(start_point.x * w), int(start_point.y * h)
        end_x, end_y = int(end_point.x * w), int(end_point.y * h)
        
        # Draw line
        cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Draw points
    for landmark in LANDMARKS_OF_INTEREST:
        idx = landmark.value
        point = landmarks.landmark[idx]
        
        # Convert normalized coordinates to pixel coordinates
        x, y = int(point.x * w), int(point.y * h)
        
        # Draw circle
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

def run_body_pose(camera_id=0, force_basic_detection=False):
    # Set paths based on operating system
    system = platform.system()
    if system == 'Darwin':  # macOS
        model_dir = 'training/models'
    else:  # Windows or other OS
        model_dir = 'models'
    
    # Model file paths
    model_path = os.path.join(model_dir, 'lstm_model.keras')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    print(f"Operating system detected: {system}")
    print(f"Using model directory: {model_dir}")
    
    if force_basic_detection:
        print("Starting body pose tracking with basic detection logic...")
        model_loaded = False
    else:
        print("Starting body pose tracking with ML model (if available)...")
        # Initialize sequence storage
        landmark_sequence = []
        
        # Load trained model and scaler
        model_loaded = False
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
            try:
                # Load the TensorFlow model
                lstm_model = tf.keras.models.load_model(model_path)
                
                # Load the scaler and label encoder
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                with open(encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                    
                model_loaded = True
                print("Loaded trained LSTM pose classification model")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using basic pose detection logic.")
        else:
            print("Trained model not found. Using basic pose detection logic.")
            print("Run 'python -m training.process_pipeline' to train a model.")
    
    # Start webcam
    cap = cv2.VideoCapture(camera_id)  # Use specified camera ID
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return False

    # Configure MediaPipe Pose with appropriate confidence thresholds
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
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
            
            # Process the image and detect pose
            results = pose.process(rgb_frame)
            
            # Draw the pose annotations on the image
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Extract gesture and display on frame
            command_detected = "No command detected"
            confidence = 0.0
            
            if results.pose_landmarks:
                # Draw pose landmarks
                if force_basic_detection:
                    # Draw full pose for basic detection
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # Draw debugging information for pose detection
                    draw_pose_debug_info(frame, results.pose_landmarks.landmark)
                else:
                    # Draw only the landmarks we're interested in for ML model
                    draw_landmarks_of_interest(frame, results.pose_landmarks)
                
                if model_loaded and not force_basic_detection:
                    # Extract features from the current frame
                    current_features = extract_pose_features(results.pose_landmarks.landmark)
                    
                    # Add to sequence
                    landmark_sequence.append(current_features)
                    
                    # Keep only the last SEQUENCE_LENGTH frames
                    if len(landmark_sequence) > SEQUENCE_LENGTH:
                        landmark_sequence.pop(0)
                    
                    # Only use the model if we have enough frames
                    if len(landmark_sequence) == SEQUENCE_LENGTH:
                        # Convert sequence to numpy array with the right shape for LSTM
                        sequence = np.array(landmark_sequence)
                        
                        # Reshape for the LSTM model (batch_size, time_steps, features)
                        n_features = sequence.shape[1]
                        
                        # Scale features using the saved scaler
                        sequence_flat = sequence.reshape(-1, n_features)
                        sequence_flat_scaled = scaler.transform(sequence_flat)
                        sequence_scaled = sequence_flat_scaled.reshape(1, SEQUENCE_LENGTH, n_features)
                        
                        # Make prediction
                        prediction_probs = lstm_model.predict(sequence_scaled, verbose=0)
                        prediction_idx = np.argmax(prediction_probs[0])
                        confidence = prediction_probs[0][prediction_idx] * 100  # Convert to percentage
                        
                        # Convert back to label
                        pose_label = label_encoder.inverse_transform([prediction_idx])[0]
                        
                        # Map the pose label to a command
                        if pose_label in POSE_MAP:
                            command_detected = POSE_MAP[pose_label]
                        else:
                            command_detected = pose_label
                    else:
                        command_detected = f"Collecting frames ({len(landmark_sequence)}/{SEQUENCE_LENGTH})"
                else:
                    # Use basic detection as fallback
                    basic_pose, basic_confidence = detect_pose_basic(results.pose_landmarks.landmark, frame)
                    if basic_pose != "unknown":
                        command_detected = basic_pose
                        confidence = basic_confidence * 100
                    
                    if confidence == 0:
                        # Calculate confidence using basic metrics of landmark visibility
                        # Base confidence from visibility scores
                        landmarks = results.pose_landmarks.landmark
                        base_confidence = np.mean([
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility
                        ])
                        confidence = base_confidence * 100  # Convert to percentage
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Display the command name on the frame
            cv2.putText(frame, f"Command: {command_detected}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display confidence if available
            if confidence > 0:
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the mode
            detection_type = "Basic Detection" if force_basic_detection or not model_loaded else "ML Model"
            cv2.putText(frame, f"Mode: Body Pose ({detection_type})", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Pose Recognition', frame)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    run_body_pose() 