#!/usr/bin/env python

""" The 'simple' goal planner subclass of GoalPlanner
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



class SimpleGoalPlanner(GoalPlanner):

	def __init__(self, robot, type_='stationary', view_distance=0.3,
					use_target_as_goal=True,goal_pose_topic=None,**kwargs):

		super(SimpleGoalPlanner, self).__init__(robot=robot,
												type_=type_,
												view_distance=view_distance,
												use_target_as_goal=use_target_as_goal,
												goal_pose_topic=goal_pose_topic)

	def find_goal_pose(self):
		"""Find a random goal pose on the map.

		Find a random goal pose within map boundaries, residing in the
		feasible pose regions associated with the map.

		Returns
		-------
		array_like
			A pose as [x,y,theta] in [m,m,degrees].
		"""
		theta = random.uniform(0, 360)

		feasible_point_generated = False
		bounds = self.feasible_layer.bounds
		#bounds = [0,5,0,4.25]

		while not feasible_point_generated:
			x = random.uniform(bounds[0], bounds[2])
			y = random.uniform(bounds[1], bounds[3])
			goal_pt = Point(x, y)
			print 'checking point: %d, %d' %(x,y)
			if self.feasible_layer.pose_region.contains(goal_pt):
				feasible_point_generated = True

		goal_pose = [x, y, theta]


		logging.info("New goal: {}".format(["{:.2f}".format(a) for a in
											goal_pose]))

		return goal_pose

	def update(self):

		super(SimpleGoalPlanner,self).update()
