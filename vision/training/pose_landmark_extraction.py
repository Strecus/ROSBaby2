import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import platform

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define the landmarks we're interested in for airplane marshalling (same as main.py)
LANDMARKS_OF_INTEREST = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]

# Set paths based on operating system
system = platform.system()
if system == 'Darwin':  # macOS
    dataset_dir = 'training/dataset'
    output_file = 'training/pose_landmarks_dataset.npz'
else:  # Windows or other OS
    dataset_dir = 'dataset'
    output_file = 'pose_landmarks_dataset.npz'

print(f"Operating system detected: {system}")
print(f"Using dataset directory: {dataset_dir}")
print(f"Output file will be saved to: {output_file}")

# Configure categories to process
categories = ['forward', 'left_turn', 'right_turn']  # As requested, focusing on these three

# Ensure output directory exists
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Store all data
all_data = []

print("Extracting pose landmarks from sequences...")

# Process each category
for category in categories:
    category_dir = os.path.join(dataset_dir, category)
    if not os.path.exists(category_dir):
        print(f"Warning: Directory {category_dir} does not exist. Skipping.")
        continue
        
    sequence_files = [f for f in os.listdir(category_dir) if f.endswith('.npz')]
    
    if not sequence_files:
        print(f"No sequence files found in {category_dir}")
        continue
        
    print(f"Processing {len(sequence_files)} sequences for category '{category}'")
    
    for seq_name in sequence_files:
        seq_path = os.path.join(category_dir, seq_name)

        # Load sequences
        try:
            data = np.load(seq_path)
            frames = data['frames']
        except Exception as e:
            print(f"Error loading {seq_path}: {e}")
            continue

        sequence_landmarks = []

        for frame in frames:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
        
            if results.pose_landmarks:
                # Extract landmarks of interest only
                landmarks = []
                for landmark_type in LANDMARKS_OF_INTEREST:
                    lm = results.pose_landmarks.landmark[landmark_type]
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                sequence_landmarks.append(landmarks)
            else:
                print(f"No pose detected in a frame from {seq_path}")
                # Use zeroes matching our reduced feature set (6 landmarks * 4 values)
                sequence_landmarks.append(np.zeros(6 * 4))

        if not sequence_landmarks:
            print(f"No landmarks extracted from {seq_path}")
            continue

        sequence_landmarks = np.array(sequence_landmarks)  # (num_frames, 6*4)

        all_data.append({
            'label': category,
            'sequence_name': seq_name,
            'landmarks': sequence_landmarks
        })

if not all_data:
    print("No pose landmarks were extracted from any sequences. Please check your input data.")
    exit(1)

print(f"Finished extracting {len(all_data)} sequences.")

# Save dataset
try:
    np.savez_compressed(output_file, data=all_data)
    print(f"Dataset saved to {output_file}")
except Exception as e:
    print(f"Error saving dataset: {e}")
    exit(1)


# Convert to pandas DataFrame
'''if all_data:
    # Create column names
    columns = ['label', 'source_sequence']
    sequence_length = len(sequence_landmarks) // (33 * 4)
    for frame_num in range(max_sequence_length):
        for i in range(33):  # 33 pose landmarks in MediaPipe
            columns.extend([f'x{frame_num}_{i}', f'y{frame_num}_{i}', f'z{frame_num}_{i}', f'v{frame_num}_{i}'])
        
    # Create and save DataFrame
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} landmark sets to {output_file}")
else:
    print("No pose landmarks were detected in any images.")'''
