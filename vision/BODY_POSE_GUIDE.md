# Body Pose Recognition Guide

This guide explains how to use the body pose recognition system for controlling the robot.

## Detection Options

The body pose mode offers two detection methods:

1. **ML Model Detection** - Uses a trained LSTM model to recognize poses based on a sequence of frames (if available)
2. **Basic Detection Logic** - Uses simple rules based on the relative positions of body landmarks

You can choose your preferred method when starting the application.

## Supported Poses

The system recognizes the following body poses:

### Arms Up - Forward Command
![Arms Up](https://i.imgur.com/placeholder-armsup.jpg)
- Raise both arms straight up above your head
- Keep your arms extended and straight
- Arms should be approximately parallel
- Used to command the robot to move forward

### Arms Down - Backward Command
![Arms Down](https://i.imgur.com/placeholder-armsdown.jpg)
- Extend both arms straight down and slightly behind you
- Keep your arms straight
- Used to command the robot to move backward

### T-Pose (Arms Out) - Stop Command
![T-Pose](https://i.imgur.com/placeholder-tpose.jpg)
- Extend both arms straight out to your sides
- Form a "T" shape with your body
- Keep your arms at shoulder height and fully extended
- Used to command the robot to stop

### Right Arm Out - Right Turn Command
![Right Arm Out](https://i.imgur.com/placeholder-rightarm.jpg)
- Extend your right arm straight out to your side
- Keep your left arm down or close to your body
- Keep your right arm fully extended
- Used to command the robot to turn right

### Left Arm Out - Left Turn Command
![Left Arm Out](https://i.imgur.com/placeholder-leftarm.jpg)
- Extend your left arm straight out to your side
- Keep your right arm down or close to your body
- Keep your left arm fully extended
- Used to command the robot to turn left

## Best Practices for Reliable Recognition

1. **Full Body Visibility**: Ensure your full upper body is visible in the camera frame.
2. **Good Lighting**: Stand in a well-lit area for better pose detection.
3. **Clear Space**: Make sure you have enough space around you to fully extend your arms.
4. **Proper Distance**: Stand about 1-3 meters from the camera.
5. **Face the Camera**: Face the camera directly for the most accurate pose detection.
6. **Distinct Poses**: Make your poses clear and distinct, avoiding subtle movements.
7. **Extended Arms**: For all poses, try to keep your arms fully extended with elbows straight.
8. **Arm Position**: Ensure your arm positions match the descriptions above precisely.

## Troubleshooting

If the system is not recognizing your poses properly:

1. Check that your full upper body is visible in the camera frame
2. Ensure there's enough lighting for the camera to see you clearly
3. Make more exaggerated, defined poses with fully extended arms
4. Check that there are no other people in the camera frame
5. Wear clothing that contrasts with the background
6. Try switching between ML model and basic detection modes
7. Restart the application if recognition becomes inconsistent

## Technical Details

The body pose recognition system uses MediaPipe's pose tracking capabilities to detect 33 landmarks on your body. These landmarks represent key joints and points such as shoulders, elbows, wrists, hips, and more.

For basic detection logic, the system focuses on the relative positions of your arms and calculates angles between shoulder, elbow, and wrist joints to determine your pose.

For ML model detection, the system captures a sequence of frames and uses a trained LSTM model to classify the pose based on all the landmark positions over time.

The confidence score for each pose is calculated differently depending on the detection method:
- For basic detection: Based on visibility of landmarks and how well your pose matches the expected position
- For ML model: Based on the model's confidence in its classification 