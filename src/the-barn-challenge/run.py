import math
import time
import numpy as np
import argparse
import subprocess
import os
from os.path import join

import rospy
import rospkg


from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import PoseWithCovarianceStamped


from gazebo_simulation import GazeboSimulation
import goalSetter

INIT_POSITION = [-2, 3, 1.57]  # in world frame
GOAL_POSITION = [-5, 10]  # relative to the initial position
def publish_initial_pose(x, y, theta):
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)
    rospy.sleep(1)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.z = math.sin(theta / 2.0)
    msg.pose.pose.orientation.w = math.cos(theta / 2.0)

    pub.publish(msg)
def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)**.5

def path_coord_to_gazebo_coord(x, y):
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5

    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift

    return (gazebo_x, gazebo_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test BARN navigation challenge')
    parser.add_argument('--world_idx', type=int, default=0)
    parser.add_argument('--gui', action="store_true")
    parser.add_argument('--out', type=str, default="out.txt")
    args = parser.parse_args()
    
    ##########################################################################################
    ## 0. Launch Gazebo Simulation
    ##########################################################################################
    
    os.environ["JACKAL_LASER"] = "1"
    os.environ["JACKAL_LASER_MODEL"] = "ust10"
    os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"
    
    if args.world_idx < 299:
        world_name = "BARN/world_%d.world" % (args.world_idx)
    else:
        world_name = "Airport/basic_airport%d.world" % (args.world_idx%299)
    
    print(f">>>>>>>>>>>>>>>> Loading Gazebo Simulation with {world_name} <<<<<<<<<<<<<<<<")
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('jackal_helper')
    os.environ['GAZEBO_PLUGIN_PATH'] = os.path.join(base_path, "plugins")
    
    launch_file = join(base_path, 'launch', 'gazebo_launch.launch')
    world_name = join(base_path, "worlds", world_name)
    
    gazebo_process = subprocess.Popen([
       'roslaunch',
        launch_file,
        'world_name:=' + world_name,
        'gui:=' + "true"
    ])
    time.sleep(5)
    


    subprocess.Popen(["gnome-terminal", "--", "roslaunch", "jackal_navigation", "amcl.launch"])

    map_file = f"/home/strecus/jackal_ws/src/jackal_navigation/maps/Airport/basic_airport{args.world_idx % 299}.yaml"
    """
    subprocess.Popen([
    "gnome-terminal", "--",
    "roslaunch", "jackal_navigation", "amcl.launch",
    f"map_file:={map_file}"
    ])
    """

    time.sleep(2)

    rospy.init_node('gym', anonymous=True)
    rospy.set_param('/use_sim_time', True)
    
    gazebo_sim = GazeboSimulation(init_position=INIT_POSITION)
    
    init_coor = (INIT_POSITION[0], INIT_POSITION[1])
    goal_coor = (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1])
    
    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)
    collided = True
    
    while compute_distance(init_coor, curr_coor) > 0.1 or collided:
        gazebo_sim.reset()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        collided = gazebo_sim.get_hard_collision()
        time.sleep(1)
    
    ##########################################################################################
    ## 1. Launch navigation stack
    ##########################################################################################
    #Se
    publish_initial_pose(INIT_POSITION[0],INIT_POSITION[1], INIT_POSITION[2])

    
    launch_file = join(base_path, '..', 'jackal_helper/launch/move_base_DWA.launch')
    nav_stack_process = subprocess.Popen(['roslaunch', launch_file])


    #gX, gY, gTheta = goalSetter.calc_goal("z.json", 1.57, INIT_POSITION[0], INIT_POSITION[1])
    #goalSetter.set_goal(-2, 10, 0)

    #goalSetter.spawn_marker(-2, 10)
        
    ##########################################################################################
    ## 2. Start navigation
    ##########################################################################################
    curr_time = rospy.get_time()
    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)
    
    while compute_distance(init_coor, curr_coor) < 0.1:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        time.sleep(0.01)
    
    start_time = curr_time
    start_time_cpu = time.time()
    collided = False
    
    while compute_distance(goal_coor, curr_coor) > .1:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        print(f"Time: {curr_time - start_time:.2f} (s), x: {curr_coor[0]:.2f} (m), y: {curr_coor[1]:.2f} (m)", end="\r")
        collided = gazebo_sim.get_hard_collision()
        while rospy.get_time() - curr_time < 0.1:
            time.sleep(0.01)
    
    ##########################################################################################
    ## 3. Report metrics and generate log
    ##########################################################################################
    
    print(">>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<")
    success = False
    if collided:
        status = "collided"
    elif curr_time - start_time >= 100:
        status = "timeout"
    else:
        status = "succeeded"
        success = True
    print(f"Navigation {status} with time {curr_time - start_time:.4f} (s)")
    
    #path_length = GOAL_POSITION[0] - INIT_POSITION[0] if args.world_idx >= 300 else np.sum([compute_distance(p1, p2) for p1, p2 in zip(path_array[:-1], path_array[1:])])
    #optimal_time = path_length / 2
    actual_time = curr_time - start_time
    
    with open(args.out, "a") as f:
        f.write(f"{args.world_idx} {success} {collided} {(curr_time - start_time)>=100} {curr_time - start_time:.4f}\n")
    
    gazebo_process.terminate()
    gazebo_process.wait()
    nav_stack_process.terminate()
    nav_stack_process.wait()

