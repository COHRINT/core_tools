#!/usr/bin/env python

""" The 'pomdp' goal planner subclass of GoalPlanner
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
import rospy

from core.robo_tools.planner import GoalPlanner
# from core.robo_tools.InterceptTestGenerator4D import InterceptTestGenerator
from core.robo_tools.belief_handling import dehydrate_msg, rehydrate_msg

from policy_translator.msg import *
from policy_translator.srv import *

class PomdpGoalPlanner(GoalPlanner):

	def __init__(self, robot, type_='stationary', view_distance=0.3,
					use_target_as_goal=True, goal_pose_topic=None, **kwargs):

		super(PomdpGoalPlanner, self).__init__(robot=robot,
												type_=type_,
												view_distance=view_distance,
												use_target_as_goal=use_target_as_goal,
												goal_pose_topic=goal_pose_topic)

		# self.policy_translator = InterceptTestGenerator() #policy translator object



	def find_goal_pose(self,positions=None):
		"""Find goal pose from POMDP policy translator server

		Parameters
		----------
		Returns
		--------
		goal_pose [array]
			Goal pose in the form [x,y,theta] as [m,m,degrees]
		"""


		msg = PolicyTranslatorRequest()
		msg.name = self.robot.name
		res = None

		if self.robot.belief is not None:
			# self.robot.belief.plot2D()
			(msg.weights,msg.means,msg.variances) = dehydrate_msg(self.robot.belief)
		else:
			msg.weights = []
			msg.means = []
			msg.variances = []

		rospy.wait_for_service('translator')
		try:
			pt = rospy.ServiceProxy('translator',policy_translator_service)
			res = pt(msg)
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e


		# print('!!!!!!!!!!!!weights:')
		# print(res.response.weights_updated)

		self.robot.belief = rehydrate_msg(res.response.weights_updated,
										res.response.means_updated,
										res.response.variances_updated)
		# print(self.robot.belief)
		# except AttributeError:
		# 	self.robot.belief = None
		# if self.robot.belief is not None:
			# self.robot.belief.plot2D()

		# print(res)

		goal_pose = res.response.goal_pose

		print(goal_pose)

		return goal_pose


		'''
		if(self.robot.name == 'Roy'):
			self.is_cop = False;
		else:
			self.is_cop = True;

		#Uncomment for the humanobservations separation
		#if(self.is_cop):
		#	self.policy_translator.getHumanObservation();

		current_position = self.robot.pose2D._pose
		for robot in positions.iteritems():
			print(positions.get(robot[0]))
			if positions.get(robot[0])[0] == 'cop':
			 	cop_pose_x = positions.get(robot[0])[1][0]
				cop_pose_y = positions.get(robot[0])[1][1]
				#self.is_cop = True
			else:
				robber_pose_x = positions.get(robot[0])[1][0]
				robber_pose_y = positions.get(robot[0])[1][1]
				#self.is_cop = False


		print('is_cop: {}, \t cop position: {} {}'.format(self.is_cop,cop_pose_x, cop_pose_y))

		xy_positions = self.policy_translator.getNextPose([cop_pose_x,cop_pose_y,robber_pose_x,robber_pose_y],self.is_cop)


		cop_orientaion = math.atan2(xy_positions[1] - cop_pose_y, xy_positions[0] - cop_pose_x)
		robber_orientaion = math.atan2(xy_positions[3] - robber_pose_y, xy_positions[2] - robber_pose_x)

		if self.is_cop:
			goal_pose = [xy_positions[0],xy_positions[1],cop_orientaion]
		else:
			goal_pose = [xy_positions[2],xy_positions[3],robber_orientaion]
		if(self.is_cop):
			print("Cop Goal");
		else:
			print("Robber Goal");
		print(goal_pose);
		#if (math.atan( (goal_pose[1]-current_position[1]) / (goal_pose[0]-current_position[0]) ) \
		#		- current_position[2]) > 120 and self.goal_pose_exsistence: #another reminder how fucking bad this is
		#	super(PomdpGoalPlanner, self).rotation_assist()
		#self.goal_pose_exsistence = True
		return goal_pose
		'''

	def update(self,positions=None):

		super(PomdpGoalPlanner,self).update(positions=positions)
