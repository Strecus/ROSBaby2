import rospy
from visualization_msgs.msg import Marker

def publish_goal_marker(x, y, z=0.0):
    rospy.init_node('goal_marker_node', anonymous=True)
    pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    marker = Marker()
    marker.header.frame_id = "map"
    marker.ns = "goal"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.3
    marker.scale.y = 0.3
    marker.scale.z = 0.3
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.lifetime = rospy.Duration(0)  # 0 = keep forever

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        pub.publish(marker)
        rospy.loginfo(f"Published marker at ({x}, {y}, {z})")
        rate.sleep()

publish_goal_marker(10.0, 2.0)
