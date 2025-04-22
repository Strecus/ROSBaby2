import math
import time
import argparse
import subprocess
import os
from os.path import join

import numpy as np
import rospy
import rospkg
from gazebo_msgs.srv import DeleteModel

from gazebo_simulation import GazeboSimulation
import goalSetter

INIT_POSITION = [-2, 3, 1.57]  # in world frame
GOAL_POSITION = [-5, 10]  # relative to the initial position

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
    
    if args.world_idx < 300:
        world_name = "BARN/world_%d.world" % (args.world_idx)
        INIT_POSITION = [-2.25, 3, 1.57]
        GOAL_POSITION = [0, 10]
    elif args.world_idx < 360:
        world_name = "DynaBARN/world_%d.world" % (args.world_idx - 300)
        INIT_POSITION = [11, 0, 3.14]
        GOAL_POSITION = [-20, 0]
    else:
        raise ValueError("World index %d does not exist" % args.world_idx)
    
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
        'gui:=' + ("true" if args.gui else "false")
    ])
    time.sleep(5)
    
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
    
    launch_file = join(base_path, '..', 'jackal_helper/launch/move_base_DWA.launch')
    nav_stack_process = subprocess.Popen(['roslaunch', launch_file])

    gX, gY, gTheta = goalSetter.calc_goal("z.json", math.pi/2)
    goalSetter.set_goal(gX, gY, gTheta)
    goalSetter.spawn_marker(gX, gY)
        
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
    
    while compute_distance(goal_coor, curr_coor) > .1 and not collided and curr_time - start_time < 100:
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

