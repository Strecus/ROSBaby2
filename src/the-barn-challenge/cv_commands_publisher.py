import json
from geometry_msgs.msg import Twist
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
    if not rospy.core.is_initialized():
        rospy.init_node('publish_goal_node', anonymous=True)
    json_path= "/home/strecus/jackal_ws/src/the-barn-challenge/"
    path = json_path + type + ".json"

    #print("Location check")
    jackal_coords = jL.location()
    #print("Location check ran with no errors")


    #print("Calc position for goal check")
    #print(jackal_coords)
    #gX, gY, gTheta = goalSetter.calc_goal(path, jackal_coords[2], jackal_coords[0], jackal_coords[1])
    #print("No errors!")

    #print("Set goal check")
    gX, gY, gTheta = jackal_coords[0], jackal_coords[1], jackal_coords[2]

    if type=="forward":
        gX += 2
    elif type=="backward":
        gX -= 2
    elif type=="right":
        gY += 2
    elif type=="left":
        gY -= 2
    goalSetter.set_goal(gX, gY, gTheta)
    #print("No errorsk")

    
    goalSetter.spawn_marker(gX, gY)




def load_path(filename="path.json"):
    """Load a recorded path from a file"""
    try:
        path = []
        with open(filename, 'r') as f:
            path = json.load(f)
        return path
        print(f"Path loaded from {filename} with {len(path)} commands.")
        print(f"Estimated path duration: {len(path) * 0.1:.1f} seconds")
    except Exception as e:
        print(f"Error loading path: {e}")
        return []




def execute_path(type, speed_factor=1.0):
    """
    Replay the recorded path
    
    Args:
        speed_factor (float): Multiplier for execution speed.
                             1.0 = original speed, 2.0 = twice as fast
    """
    path = load_path("/home/strecus/jackal_ws/src/the-barn-challenge/"+type+".json")
    rospy.init_node('cmd_vel_path_follower')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    rate = rospy.Rate(10 * speed_factor)  # 10Hz times speed factor


    twist = Twist() 
    
    if not path:
        print("No path to execute. Record a path first.")
        return
    print(path)
    print(f"Executing path with {len(path)} commands...")
    
    try:
        for i, cmd in enumerate(path):
            linear, angular, duration = cmd

            # Set movement commands
            twist.linear.x = linear
            twist.angular.z = angular

            movement_time = duration / speed_factor  # seconds
            publish_rate = 10  # 10 Hz
            rate = rospy.Rate(publish_rate)

            num_publish = int(movement_time * publish_rate)
            for _ in range(num_publish):
                pub.publish(twist)
                rate.sleep()



            # Log progress occasionally
            if i % 20 == 0 or i == len(path) - 1:
                print(f"Executing command {i+1}/{len(path)}...")

        # Ensure the robot stops at the end
        twist.linear.x = 0
        twist.angular.z = 0
        pub.publish(twist)
        print("Path execution complete.")

    except Exception as e:
        # Safety stop if anything goes wrong
        twist.linear.x = 0
        twist.angular.z = 0
        pub.publish(twist)
        print(f"Error during path execution: {e}")





def main():
    #rospy.init_node("goal_queuer", anonymous=True)

    # Queue up multiple goals
    execute_path("forward")
   




   

if __name__ == "__main__":
    main()


