import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped

# Global variable to store the position
jackal_coords = None

def pose_callback(msg):
    global jackal_coords
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    theta = msg.pose.pose.orientation.z  # Assuming a simple 2D setup
    jackal_coords = (x, y, theta)

def get_jackal_position_using_amcl():
    """
    Get the current position of the Jackal robot using AMCL's /amcl_pose topic.
    Returns a tuple (x, y, theta) for the robot's position and orientation.
    """
    global jackal_coords

    #rospy.init_node('get_jackal_position_using_amcl', anonymous=True)

    # Subscribe to the /amcl_pose topic
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, pose_callback)

    # Wait until we get the first valid position
    while jackal_coords is None:
        rospy.sleep(0.1)  # Wait for the message to arrive

    # Optionally, log the position once it's retrieved
    rospy.loginfo(f"Jackal position: {jackal_coords}")

    # Shutdown the node after getting the position
    #rospy.signal_shutdown("Position retrieved")

    return jackal_coords

# Example usage in the main function
if __name__ == "__main__":
    # Calling the method to get Jackal's position
    position = get_jackal_position_using_amcl()
    if position:
        x, y, theta = position
        rospy.loginfo(f"Current Position: x={x}, y={y}, theta={theta}")
