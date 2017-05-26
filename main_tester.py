#!/usr/bin/env python
import matplotlib
matplotlib.use('Qt4Agg')

import logging
import time
import sys
import os

import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

from core.helpers.config import load_config
from core.robo_tools.vagabond import Vagabond
from core.robo_tools.cop import Cop
from core.robo_tools.robber import Robber

class ReliableVagabond(object):
	"""
	Parameters
	----------
	param : param_type, optional
		param_description
	Attributes
	----------
	attr : attr_type
		attr_description
	Methods
	----------
	attr : attr_type
		attr_description
	"""

	def __init__(self, config_file='config.yaml'):

		# Load configuration files
		self.cfg = load_config(config_file)

	   # Configure Python's logging
		logger_level = logging.getLevelName(self.cfg['main']['logging_level'])
		logger_format = '[%(levelname)-7s] %(funcName)-30s %(message)s'
		try:
			logging.getLogger().setLevel(logger_level)
			logging.getLogger().handlers[0]\
				.setFormatter(logging.Formatter(logger_format))
		except IndexError:
			logging.basicConfig(format=logger_format,
								level=logger_level,
							   )
		np.set_printoptions(precision=self.cfg['main']['numpy_print_precision'],suppress=True)
		# Set up a ROS node (if using ROS)
		if self.cfg['main']['use_ROS']:
			import rospy
			rospy.init_node(self.cfg['main']['ROS_node_name'],
							log_level=rospy.DEBUG)

		# Link node to Python's logger
		handler = logging.StreamHandler()
		handler.setFormatter(logging.Formatter(logger_format))
		logging.getLogger().addHandler(handler)

		# robot positions
		self.positions = {}

		# Create robots
		self.create_actors()
		time.sleep(4)
		self.headless_mode()

	#Start from here

	def headless_mode(self):
		"""Runs the simulation without any animation output.
		"""
		i = 0
		max_run_time = self.cfg['main']['max_run_time']
		while i < max_run_time:
			self.update(i)
			i += 1

		logging.warn('Experiment has reached the max run time of {} frames! \
						\nExperiment ending...'.format(i))
		# while self.vagabonds['Deckard'].mission_planner.mission_status != 'stopped':
		#	 self.update(i)
		#	 i += 1
			# time.sleep(0.2)

	def update(self, i):
		"""Update all the major aspects of the simulation and record data.
		"""
		logging.debug('Main update frame {}'.format(i))
		# Update all actors
		for robot_name, robot in self.robots.iteritems():
			robot.update(i,self.positions)
			tmpKey = self.positions[robot_name];
			tmpKey[1] = robot.pose2D._pose;
		#	print('************************TMPKEY***********************************')
			#print(tmpKey);


			self.positions[robot_name] = tmpKey;

			logging.debug('{} update'.format(robot_name))

	def create_actors(self):
		self.robots = {}

		for robot, kwargs in self.cfg['robots'].iteritems():
			if self.cfg['robots'][robot]['use']:
				if self.cfg['robots'][robot]['type_'] == 'vagabond':
					self.robots[robot] = Vagabond(robot, **kwargs)
				elif self.cfg['robots'][robot]['type_'] == 'cop':
					self.robots[robot] = Cop(robot, **kwargs)
					self.positions[robot] = [self.cfg['robots'][robot]['type_'],(1,1,0)]
				elif self.cfg['robots'][robot]['type_'] == 'robber':
					self.robots[robot] = Robber(robot, **kwargs)
					self.robots[robot].type_ = 'robber'
					self.positions[robot] = [self.cfg['robots'][robot]['type_'],(3,3,0)]
				logging.info('{} added to simulation'.format(robot))
				#self.positions[robot] = [self.cfg['robots'][robot]['type_'],(0,0,0)]

		# Create robbers with config params

		# Use Deckard's map as the main map
		#self.map = self.vagabonds['Deckard'].map

if __name__ == '__main__':
	rv = ReliableVagabond()
