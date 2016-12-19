#!/usr/bin/env python

# ROS-based pose feasibility function

"""
map_sever will publish occupancy grid as an OccupancyGrid.msg message.
This message contains a header, a MetaData message, and a 2D array of
occupancy values (0 to 100).

The MetaData message, MapMetaData.msg, contains resolution (m/cell),
width (cells), height (cells), and the origin as a Pose.msg

The topic, /map/, is a latched topic, which means messages published are
sent once to each new subscriber. The messages are also available via sevice.
------------
Parameters:
    goal_pose [m,m,deg]

Return:
    feasibility: bool
------------
- goal_pose coordinates [m,m] * 1/resolution [cells/m] = goal_pose [cells]
- compare with occupancy grid cells
- return feasibiltiy

http://wiki.ros.org/map_server
"""

#<>TODO: add subsriber node to robot.py to receive occupancy grid!

def feasible_pose(goal_pose,*args):
    resolution = map_resolution
    robot_diameter = robot_diameter_
    grid_area_dim = int((robot_diameter/2) / resolution) + 1 #1/2 of side of square

    goal_pose_cells = [int(goal_pose[0]*(1/resolution)),
                        int(goal_pose[1]*(1/resolution))]

    if goal_pose_cells[0] - grid_area_dim > 0:
        x_range_min = goal_pose_cells[0] - grid_area_dim
    else:
        x_range_min = 0
    if goal_pose_cells[1] - grid_area_dim > 0:
        y_range_min = goal_pose_cells[1] - grid_area_dim
    else:
        y_range_min = 0
    if goal_pose_cells[0] + grid_area_dim < map_width:
        x_range_max = goal_pose_cells[0] + grid_area_dim
    else:
        x_range_max = map_width
    if goal_pose_cells[1] + grid_area_dim < map_height:
        y_range_max = goal_pose_cells[1] + grid_area_dim
    else:
        y_range_max = map_height

    print(x_range_min,x_range_max)
    print(y_range_min,y_range_max)

    for x_cell in range(x_range_min,x_range_max):
        for y_cell in range(y_range_min,y_range_max):
            k = map_width * (x_cell) + (y_cell)
            print(occupancy_grid[k])
            if occupancy_grid[k] > occupancy_threshold:
                return False

    return True

map_resolution = 1
robot_diameter_ = 1
occupancy_threshold = 50

map_width = int(5/map_resolution)
map_height = int(5/map_resolution)

# occupancy_grid = [ [70,70,0,70,70],
#                     [70,0,0,0,70],
#                     [0,0,0,0,0],
#                     [0,0,70,0,0],
#                     [0,0,70,0,0] ]

occupancy_grid = [70,70,0,70,70,70,0,0,0,70,0,0,0,0,0,0,0,70,0,0,0,0,70,0,0]

"""
occupancy_grid = [ [70,70,0,70,70,0,0,0,0,0],
                    [70,0,0,0,70,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,70,0,0,0,0,0,0,0],
                    [0,0,70,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0] ]
"""

goal_poses = [ [2,3],
                [0,1],
                [5,5],
                [2,2],
                [4,1] ]

# args = {"map_resolution": map_resolution,
#             "robot_diameter": robot_diameter,
#             "map_width": map_width,
#             "map_height": map_height,
#             "occupancy_grid": occupancy_grid,
#             "occupancy_threshold": occupancy_threshold}

for pose in goal_poses:
    feasible = feasible_pose(pose,map_resolution,robot_diameter_,map_width,map_height,
                occupancy_grid,occupancy_threshold)
    print(pose,feasible)
