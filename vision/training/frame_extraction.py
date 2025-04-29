import cv2
import os
import numpy as np
import platform

# Configuration 
video_dirs = ['forward', 'left_turn', 'right_turn']  # Focusing on these three as requested

# Set paths based on operating system
system = platform.system()
if system == 'Darwin':  # macOS
    base_video_dir = 'training/videos'
    base_output_dir = 'training/dataset'
else:  # Windows or other OS
    base_video_dir = 'videos'
    base_output_dir = 'dataset'

print(f"Operating system detected: {system}")
print(f"Using video directory: {base_video_dir}")
print(f"Using output directory: {base_output_dir}")

desired_fps = 10  # Extract 10 frames per second 
sequence_length = 10 #Sequence of 10 frames for each sample

# Process each category
for category in video_dirs:
    video_dir = os.path.join(base_video_dir, category)
    output_dir = os.path.join(base_output_dir, category)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create video directory if it doesn't exist
    if not os.path.exists(video_dir):
        print(f"Creating video directory: {video_dir}")
        os.makedirs(video_dir, exist_ok=True)
        print(f"⚠️ Please add video files to {video_dir}")
        continue
    
    # Get all video files in this category
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"⚠️ No video files found in {video_dir}")
        continue
        
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / desired_fps)  # Extract frames at desired rate
        
        frame_count = 0
        saved_frame_count = 0
        frame_sequence = [] # Holds frames for current sequence
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_sequence.append(frame) #Add frame to the current sequence

                #Check if we have enough frames for a sequence:
                if len(frame_sequence) == sequence_length:
                    #Save the sequence as one sample

                # Create a unique filename for each frame -> sequence
                    sequence_filename = os.path.join(
                        output_dir, 
                        f"{os.path.splitext(video_file)[0]}_seq_{saved_frame_count:04d}.npz"
                    )
                    #Save frames as numpy array
                    #cv2.imwrite(frame_filename, frame)
                    np.savez(sequence_filename, frames = frame_sequence)
                    saved_frame_count += 1

                    #Clear for next sequence
                    frame_sequence = []
                    
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_frame_count} frames from {video_file}")

print("Frame extraction complete!")