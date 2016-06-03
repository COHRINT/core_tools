#!/usr/bin/env python
import matplotlib
matplotlib.use('Qt4Agg')

import logging
import time
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core.helpers.config import load_config
from core.robo_tools.vagabond import Vagabond

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

                    # Create robots
                    self.create_actors()


        # Describe the simulation
        logging.info(self)

        self.headless_mode()

    #def __repr__ ??????????????????????
    #Start from here

    def headless_mode(self):
        """Runs the simulation without any animation output.
        """
        i = 0
        while self.vagabond['Deckard'].mission_planner.mission_status != 'stopped':
            self.update(i)
            i += 1

    def update(self, i):
        """Update all the major aspects of the simulation and record data.
        """
        logging.debug('Main update frame {}'.format(i))

        # Update all actors
        for robot_name, vagabond in self.vagabond.iteritems():
            vagabond.update(i)

    def create_actors(self):
        # Create robbers with config params
        other_robot_names = {'vagabonds': []}
        self.vagabonds = {}
        i = 0
        for vagabond, kwargs in self.cfg['robots'].iteritems():
            if i >= self.cfg['main']['number_of_agents']['robots']:
                break
            self.vagabonds[robber] = Robber(robot, **kwargs)
            logging.info('{} added to simulation.'.format(robots))
            i += 1
            other_robot_names['robbers'].append(robber)

        # Use Deckard's map as the main map
        self.map = self.cops['Deckard'].map

        # <>TODO: Replace with message passing, potentially
        # Give cops references to the robber's actual poses
        for cop in self.cops.values():
            for robber_name, robber in self.robbers.iteritems():
                cop.missing_robbers[robber_name].pose2D = robber.pose2D
            for distrator_name, distractor in self.distractors.iteritems():
                cop.distracting_robots[distrator_name].pose2D = \
                    distractor.pose2D

        # Give robbers the list of found robots, so they will stop when found
        for robber in self.robbers.values():
            robber.found_robbers = self.cops['Deckard'].found_robbers

if __name__ == '__main__':
    cnr = ReliableVagabond()