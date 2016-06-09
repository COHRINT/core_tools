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

        self.headless_mode()

    #Start from here

    def headless_mode(self):
        """Runs the simulation without any animation output.
        """
        i = 0
        # Works because Deckard is one of the robots in the config file
        while self.vagabond['Deckard'].mission_planner.mission_status != 'stopped':
            self.update(i)
            i += 1

    def update(self, i):
        """Update all the major aspects of the simulation and record data.
        """
        logging.debug('Main update frame {}'.format(i))

        # Update all actors
        for vagabond_name, vagabond in self.vagabond.iteritems():
            vagabond.update(i)

    def create_actors(self):
        self.vagabonds = {}

        for vagabond in self.cfg['number of agents']['vagabonds']:
            self.vagabonds[vagabond] = Vagabond(vagabond)

        # Create robbers with config params
        
        # Use Deckard's map as the main map
        #self.map = self.vagabonds['Deckard'].map

if __name__ == '__main__':
    rv = ReliableVagabond()