#!/usr/bin/env python

""" The 'trajectory' goal planner subclass of GoalPlanner
"""

__author__ = ["Ian Loefgren", "Sierra Williams"]
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "2.0.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

import random
import logging
import math
from shapely.geometry import Point, LineString
from core.robo_tools.planner import GoalPlanner

class TrajectoryGoalPlanner(GoalPlanner):

	def __init__(self, robot, type_='stationary', view_distance=0.3,
                    use_target_as_goal=True, goal_pose_topic=None, **kwargs):

        super(TrajectoryGoalPlanner, self).__init__(robot=robot,
                                                type_=type_,
                                                view_distance=view_distance,
                                                use_target_as_goal=use_target_as_goal,
                                                goal_pose_topic=goal_pose_topic)

	def find_goal_pose(self):
        """Find a goal pose from a set trajectory, defined by the
            mission_planner.
            Trajectory must be a iterated numpy array

        Parameters
        ----------
        Returns
        -------
        array_like
            A pose as [x,y,theta] in [m,m,degrees].

        """
        trajectory = self.robot.mission_planner.trajectory
        try:
            next_goal = next(trajectory)
        except NameError:
            logging.warn('Trajectory not found')
            return None
        except StopIteration:
            logging.info('The specified trajectory has ended')
            return None
        except:
            logging.warn('Unknown Error in loading trajectory')
            return None

        if next_goal.size == 2:
            # if theta is not specified, don't rotate
            current_pose = self.robot.pose2D.pose[0:2]
            theta = math.atan2(next_goal[1] - current_pose[1],
                               next_goal[0] - current_pose[0])  # [rad]
            theta = math.degrees(theta) % 360
            goal_pose = np.append(next_goal, theta)
        else:
            goal_pose = next_goal

        return goal_pose

    def update(self):

        super(TrajectoryGoalPlanner,self).update()
