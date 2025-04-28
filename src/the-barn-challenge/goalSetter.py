

#!/usr/bin/env python3


import math
import rospy 
import actionlib
from gazebo_msgs.srv import DeleteModel
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
import tf
import json
import subprocess


def spawn_marker(gX, gY):
    # Spawn new model at goal location
    
    try:
        rospy.wait_for_service('/gazebo/delete_model', timeout=2)
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model('goal_marker')
        rospy.loginfo("Deleted old goal_marker from Gazebo.")
    except rospy.ServiceException as e:
        rospy.logwarn("goal_marker could not be deleted or didn't exist: %s", str(e))
    except rospy.ROSException:
        rospy.logwarn("Timeout waiting for /gazebo/delete_model")

    rospy.sleep(0.5)  #Gazebo is a little baby boi


    try:
        rospy.sleep(0.5)
        subprocess.run([
            "rosrun", "gazebo_ros", "spawn_model",
            "-file", "/home/strecus/jackal_ws/src/viz/marker_model.sdf",
            "-sdf",
            "-model", "goal_marker",
            "-x", str(gX),
            "-y", str(gY),
            "-z", "0"
        ], check=True)
        rospy.loginfo("Spawned goal_marker in Gazebo.")
    except subprocess.CalledProcessError as e:
        rospy.logerr("Failed to spawn model: %s", str(e))

        
def calc_goal(path="z.json",intialTheta=0.0, intialX=0.0, intialY=0.0):
    

    if not path:
        print("No path to execute. Record a path first.")
        return None

    gX, gY, gTheta =intialX, intialY, intialTheta  # Global pose
    

    pathArr = []
    with open(path, 'r') as f:
        pathArr = json.load(f)
   

    try:
        for i, cmd in enumerate(pathArr):

            linear = cmd[0]
            w = cmd[1]
            duration = cmd[2]


            gX += linear * duration * math.cos(gTheta)
            gY += linear * duration * math.sin(gTheta)


            #if linear != 0 and angular == 0:
            if not abs(w) < .00001:
               gTheta += w * duration
    
        #Keep in range of a circle
        gTheta = math.atan2(math.sin(gTheta), math.cos(gTheta))



        return gX, gY, gTheta

    except Exception as e:
        # Safety stop if anything goes wrong
        print(f"Error during goal calulation with path: {e}")
        return None

def set_goal(gX, gY, gTheta):
   


    if gX is not None and gY is not None and gTheta is not None:
        print(f"Goal set to: X={gX}, Y={gY}, Theta={gTheta}")

        #Converts theta yaw into a quaternion which move_base needs
        gQ = tf.transformations.quaternion_from_euler(0, 0, gTheta)

        #Estavlishes the server that this goal will be sent to
        client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        # Lets the move_base node recieve the goal
        

        
        client.wait_for_server()

        goal_goal = MoveBaseGoal()
        goal_goal.target_pose.header.frame_id = "map"
        goal_goal.target_pose.header.stamp = rospy.Time.now()

        goal_goal.target_pose.pose.position.x = gX
        goal_goal.target_pose.pose.position.y = gY

        goal_goal.target_pose.pose.orientation = Quaternion(*gQ)

        rospy.loginfo(f"Sending goal to ({gX:.2f}, {gY:.2f}, θ={gTheta:.2f} rad)")

        client.send_goal(goal_goal)
        


    else:
        print("Failed to set goal.")



def main():
    
    print("Hello, World!")

if __name__ == "__main__":
    main()
