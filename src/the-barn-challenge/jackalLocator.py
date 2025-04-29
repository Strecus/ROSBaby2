#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
import tf
import math

def location():
    """Get the true x, y, yaw (radians) of the Jackal from Gazebo."""
    
    # Wait until model states is available
    data = rospy.wait_for_message('/gazebo/model_states', ModelStates)
    
    try:
        idx = data.name.index('jackal')
    except ValueError:
        rospy.logerr("Jackal not found in /gazebo/model_states!")
        return None, None, None
    
    pose = data.pose[idx]
    x = pose.position.x
    y = pose.position.y
  
    # Convert quaternion to yaw
    orientation = pose.orientation
    (_, _, yaw) = tf.transformations.euler_from_quaternion([
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w
    ])
    print("Jackal's position is: (" + x+"," + y+"," +z+")")
    return x, y, yaw

if __name__ == "__main__":
    rospy.init_node('jackal_ground_truth_reader', anonymous=True)

    x, y, yaw = get_true_jackal_pose()

    if x is not None:
        print(f"Jackal True Position:")
        print(f"  X: {x:.3f} meters")
        print(f"  Y: {y:.3f} meters")
        print(f"  Yaw: {yaw:.3f} radians")
