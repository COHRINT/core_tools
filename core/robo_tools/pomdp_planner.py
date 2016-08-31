#!/usr/bin/env python

""" The 'pomdp' goal planner subclass of GoalPlanner
"""
import random
import logging
import math
from shapely.geometry import Point, LineString
from core.robo_tools.planner import GoalPlanner
from core.robo_tools.testPolicyTranslator import testPolicyTranslator

class PomdpGoalPlanner(GoalPlanner):

	def __init__(self, robot, type_='stationary', view_distance=0.3,
					use_target_as_goal=True, goal_pose_topic=None, **kwargs):

		super(PomdpGoalPlanner, self).__init__(robot=robot,
												type_=type_,
												view_distance=view_distance,
												use_target_as_goal=use_target_as_goal,
												goal_pose_topic=goal_pose_topic)

		self.policy_translator = testPolicyTranslator() #policy translator object

	def find_goal_pose(self):
		"""Find goal pose from POMDP policy

		Parameters
		----------
		Returns
		--------
		goal_pose [array]
			Goal pose in the form [x,y,theta] as [m,m,degrees]
		"""
		current_position = self.robot.pose2D._pose

		goal_pose = self.policy_translator.getNextPose(current_position)

		return goal_pose

	def update(self):

		super(PomdpGoalPlanner,self).update()
