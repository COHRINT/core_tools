#!/usr/bin/env python
"""Generic definition of a robot.

Currently subclasses from the iRobotCreate class to allow for
physical control of an iRobot Create base (if the Robot class is
configured to control hardware) but could be subclassed to use other
physical hardware in the future.

A robot has a planner that allows it to select goals and a map to
keep track of other robots, feasible regions to which it can move,
an occupancy grid representation of the world, and role-specific
information (such as a probability layer for the rop robot to keep
track of where robber robots may be).

"""
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "2.0.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

import logging
import math
import random
import numpy as np

from shapely.geometry import Point, Polygon

from abc import ABCMeta, abstractmethod

from core.robo_tools.pose import Pose
from core.robo_tools.planner import (MissionPlanner,
                                             GoalPlanner,
                                             PathPlanner,
                                             Controller)
from core.map_tools.map import Map
from core.map_tools.map_elements import MapObject


class Robot(object):
    __metaclass__ = ABCMeta
    """Class definition for the generic robot object.

    .. image:: img/classes_Robot.png

    Parameters
    ----------
    name : str
        The robot's name.
    pose : array_like, optional
        The robot's initial [x, y, theta] in [m,m,degrees] (defaults to
        [0, 0.5, 0]).
    pose_source : str
        The robots pose source. Either  a rostopic name, like 'odom' or
        'tf', or 'python'
    color_str : str
        The color of the robots map object
    **kwargs
        Arguments passed to the ``MapObject`` attribute.

    """

    def __init__(self,
                 name,
                 pose=None,
                 pose_source='python',
                 color_str='darkorange',
                 map_cfg={},
                 create_mission_planner=True,
                 goal_planner_cfg={},
                 path_planner_cfg={},
                 **kwargs):

        # Object attributes
        self.name = name
        self.pose_source = pose_source

        # Robot belief
        # self.belief = None

        # Setup map
        self.map = Map(**map_cfg)

        # If pose is not given, randomly place in feasible layer.
        feasible_robot_generated = False
        if pose is None:
            while not feasible_robot_generated:
                x = random.uniform(self.map.bounds[0], self.map.bounds[2])
                y = random.uniform(self.map.bounds[1], self.map.bounds[3])
                if self.map.feasible_layer.pose_region.contains(Point([x, y])):
                    feasible_robot_generated = True
            theta = random.uniform(0, 359)
            pose = [x, y, theta]

        self.pose2D = Pose(self, pose, pose_source)
        self.pose_history = np.array(([0, 0, 0], self.pose2D.pose))
        if pose_source == 'python':
            self.publish_to_ROS = False
        else:
            self.publish_to_ROS = True

        # if publishing to ROS, create client for occupancy grid service
        if self.publish_to_ROS:
            import rospy
            import nav_msgs.msg
            from nav_msgs.msg import OccupancyGrid, MapMetaData
            from nav_msgs.srv import GetMap
            rospy.wait_for_service('/' + self.name.lower() + '/static_map')
            try:
                get_map = rospy.ServiceProxy('/' + self.name.lower() + '/static_map',GetMap)
                map_msg = get_map()
                logging.info("Received new map")
            except rospy.ServiceException, e:
                print "Service call for map failed: %s"%e

            self.map_server_info_update(map_msg)

        # Setup planners
        if create_mission_planner:
            self.mission_planner = MissionPlanner(self)

        goal_planner_type = goal_planner_cfg['type_']

        if goal_planner_type == 'stationary':
           target_pose = None

        elif goal_planner_type == 'simple':
            from simple_planner import SimpleGoalPlanner
            self.goal_planner = SimpleGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'trajectory':
            from trajectory_planner import TrajectoryGoalPlanner
            self.goal_planner = TrajectoryGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'particle':
            from particle_planner import ParticleGoalPlanner
            self.goal_planner = ParticleGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'MAP':
            from probability_planner import PorbabilityGoalPlanner
            self.goal_planner = ProbabilityGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'pomdp':
            from pomdp_planner import PomdpGoalPlanner
            self.goal_planner = PomdpGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'audio':
            from audio_planner import AudioGoalPlanner
            self.goal_planner = AudioGoalPlanner(self,**goal_planner_cfg)

        # elif self.goal_planner_type == 'trajectory':
        #     self.goal_planner
        # elif self.type == 'particle':
        #     target_pose = self.find_goal_from_particles()
        # elif self.type == 'MAP':
        #     target_pose = self.find_goal_from_probability()
        #
        # self.goal_planner = GoalPlanner(self,
        #                                 **goal_planner_cfg)
        # If pose_source is python, this robot is just in simulation
        if not self.publish_to_ROS:
            self.path_planner = PathPlanner(self, **path_planner_cfg)
            self.controller = Controller(self)

        # Define MapObject
        # <>TODO: fix this horrible hack
        create_diameter = 0.34
        self.diameter = create_diameter
        if self.name == 'Deckard':
            pose = [0, 0, -np.pi / 4]
            r = create_diameter / 2
            n_sides = 4
            pose = [0, 0, -np.pi / 4]
            x = [r * np.cos(2 * np.pi * n / n_sides + pose[2]) + pose[0]
                 for n in range(n_sides)]
            y = [r * np.sin(2 * np.pi * n / n_sides + pose[2]) + pose[1]
                 for n in range(n_sides)]
            shape_pts = Polygon(zip(x, y)).exterior.coords
        else:
            shape_pts = Point([0, 0]).buffer(create_diameter / 2)\
                .exterior.coords
        self.map_obj = MapObject(self.name, shape_pts[:], has_relations=False,
                                 blocks_camera=False, color_str=color_str)

        self.update_shape()

    def map_server_info_update(self,occupancy_grid_msg):
        """Update stored info about occupancy_grid
        """
        self.occupancy_grid = occupancy_grid_msg.map.data
        self.map_resolution = occupancy_grid_msg.map.info.resolution
        self.map_height = occupancy_grid_msg.map.info.height
        self.map_width = occupancy_grid_msg.map.info.width
        logging.info("Map metadata updated")

    def update_shape(self):
        """Update the robot's map_obj.
        """
        self.map_obj.move_absolute(self.pose2D.pose)

    @abstractmethod
    def update(self, i=0, positions=None):
        """Update all primary functionality of the robot.

        This includes planning and movement for both cops and robbers,
        as well as sensing and map animations for cops.

        Parameters
        ----------
        i : int, optional
            The current animation frame. Default is 0 for non-animated robots.


        """
        # <>TODO: @Matt Figure out how to move this back to pose class.
        if self.pose_source == 'tf':
            self.pose2D.tf_update()

        if self.mission_planner.mission_status is not 'stopped':
            # Update statuses and planners
            self.mission_planner.update()
            self.goal_planner.update(positions=positions)
            if self.publish_to_ROS is False:
                self.path_planner.update()
                self.controller.update()

            # Add to the pose history, update the map
            self.pose_history = np.vstack((self.pose_history,
                                           self.pose2D.pose[:]))
            self.update_shape()



class ImaginaryRobot(object):
    """An imaginary robot
        This robot has a name, and can be given attributes.
        Useful for belief representation.
    """
    def __init__(self, name):
        self.name = name
