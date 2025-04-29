## Airplane Marshalling CV Robot with Hand Gestures and Body Poses

## Overview
This project implements a computer vision-based system that recognizes both hand gestures and body poses for robot control. The system offers two modes of operation:

1. **Hand Gesture Recognition** - Uses MediaPipe Hands to recognize hand gestures for intuitive control
2. **Body Pose Recognition** - Uses MediaPipe Pose and optional LSTM model to recognize full body postures

The user can choose the desired recognition mode when starting the application.

## Available Commands

Both recognition modes support the same set of five commands:

- **Forward**: Move the robot forward
- **Backward**: Move the robot backward
- **Left**: Turn the robot left
- **Right**: Turn the robot right
- **Stop**: Stop the robot's movement

## Hand Gestures
When using hand gesture mode, the system recognizes the following gestures:

- **Thumb Up**: Forward command
- **Thumb Down**: Backward command
- **Open Palm**: Stop command
- **Pointing Left**: Left turn command
- **Pointing Right**: Right turn command

See `HAND_GESTURE_GUIDE.md` for detailed instructions on performing these gestures.

## Body Poses
When using body pose mode, the system recognizes the following poses:

- **Arms Up**: Forward command
- **Arms Down**: Backward command
- **T-Pose (arms out to sides)**: Stop command
- **Left Arm Out**: Left turn command
- **Right Arm Out**: Right turn command

See `BODY_POSE_GUIDE.md` for detailed instructions on performing these poses.

## Setup

1. Clone this repo `git clone https://github.com/amahjoor/CS485_Project.git` (in directory you want to save it) 
2. Create and activate your virtual environment:
```bash
# for Windows
python -m venv venv
venv\Scripts\activate

# for MacOS
python -m venv venv
source venv/bin/activate
```
3. Download videos and upload them to training/videos 
4. Run the process pipeline (installation, extraction, training) `python process_pipeline.py`
5. Run the application: `python main.py`
