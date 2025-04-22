
import rospy
import subprocess
import actionlib
from move_base_msgs.msg import MoveBaseAction
import goalSetter

def wait_for_goal_to_finish():
    client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    client.wait_for_server()
    client.wait_for_result()

def publish_goal(path="z.json"):
    gX, gY, gTheta = goalSetter.calc_goal(path)
    goalSetter.set_goal(gX, gY, gTheta)
        
    
    goalSetter.spawn_marker(gX, gY)

def main():
    rospy.init_node("goal_queuer", anonymous=True)

    # Queue up multiple goals
    publish_goal("a.json")
    wait_for_goal_to_finish()

   

if __name__ == "__main__":
    main()
