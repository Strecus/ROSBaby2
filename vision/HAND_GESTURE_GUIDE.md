# Hand Gesture Recognition Guide

This guide explains how to use the hand gesture recognition system for controlling the robot.

## Supported Gestures

The system recognizes the following hand gestures:

### Thumb Up - Forward Command
![Thumb Up](https://i.imgur.com/placeholder-thumbup.jpg)
- Extend your thumb upward
- Keep other fingers closed
- Used to command the robot to move forward

### Thumb Down - Backward Command
![Thumb Down](https://i.imgur.com/placeholder-thumbdown.jpg)
- Extend your thumb downward
- Keep other fingers closed
- Used to command the robot to move backward

### Open Palm - Stop Command
![Open Palm](https://i.imgur.com/placeholder-openpalm.jpg)
- Extend all fingers upward with your palm facing the camera
- Shows all fingers extended and visible
- Used to command the robot to stop

### Pointing Right - Right Turn Command
![Pointing Right](https://i.imgur.com/placeholder-pointright.jpg)
- Extend your index finger horizontally to the right
- The finger should point more horizontally than vertically
- Used to command the robot to turn right

### Pointing Left - Left Turn Command
![Pointing Left](https://i.imgur.com/placeholder-pointleft.jpg)
- Extend your index finger horizontally to the left
- The finger should point more horizontally than vertically
- Used to command the robot to turn left

## Best Practices for Reliable Recognition

1. **Good Lighting**: Ensure your hand is well-lit for better recognition.
2. **Proper Distance**: Keep your hand at a comfortable distance from the camera (about 30-60 cm).
3. **Clear Background**: A clean, uncluttered background helps with detection.
4. **Hand Position**: Face your palm toward the camera for better landmark detection.
5. **Distinct Gestures**: Make your gestures clear and distinct.

## Troubleshooting

If the system is not recognizing your gestures properly:

1. Check the lighting in your environment
2. Make sure your hand is fully visible in the camera frame
3. Try adjusting the distance between your hand and the camera
4. Ensure your gestures are clear and match the descriptions above
5. Restart the application if recognition becomes inconsistent

## Technical Details

The hand gesture recognition system uses MediaPipe's hand tracking capabilities to detect 21 landmarks on each hand. These landmarks are the key points on your hand (fingertips, knuckles, wrist, etc.) that the system uses to determine the position and orientation of your fingers.

The relative positions of these landmarks are analyzed to recognize specific hand gestures according to predefined rules. For example, a thumb-up gesture is detected when the thumb tip is positioned above the wrist while the other fingers are closed.

For the pointing gestures (left and right), the system tracks the horizontal position of your index finger relative to its base knuckle. When pointing left, your index finger tip should be to the left of its base; when pointing right, it should be to the right. 