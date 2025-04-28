
import rospy
import subprocess
import actionlib
from move_base_msgs.msg import MoveBaseAction
import goalSetter


import sys; sys.path.append('/home/strecus/jackal_ws/src/the-barn-challenge'); import jackalLocator as jL


def wait_for_goal_to_finish():
    client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    client.wait_for_server()
    client.wait_for_result()

def publish_goal(type="forward"):
    json_path= "/home/strecus/jackal_ws/src/the-barn-challenge/"
    path = json_path + type + ".json"

    jackal_coords = jL.get_jackal_position_using_amcl()
    
    gX, gY, gTheta = goalSetter.calc_goal(path, jackal_coords[2], jackal_coords[0], jackal_coords[1])
    goalSetter.set_goal(gX, gY, gTheta)
     
    
    goalSetter.spawn_marker(gX, gY)

def main():
    rospy.init_node("goal_queuer", anonymous=True)

    # Queue up multiple goals
    publish_goal("forward")
    wait_for_goal_to_finish()

   

if __name__ == "__main__":
    main()
