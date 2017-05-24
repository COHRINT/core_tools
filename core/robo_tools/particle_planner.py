#!/usr/bin/env python

""" The 'particle' goal planner subclass of GoalPlanner
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

class ParticleGoalPlanner(GoalPlanner):

    def __init__(self, robot, type_='stationary', view_distance=0.3,
                    use_target_as_goal=True, goal_pose_topic=None, **kwargs):

        super(ParticleGoalPlanner, self).__init__(robot=robot,
                                                type_=type_,
                                                view_distance=view_distance,
                                                use_target_as_goal=use_target_as_goal,
                                                goal_pose_topic=goal_pose_topic)

	def find_goal_pose(self):
        """Find a goal from the most likely particle(s).

        Find a goal pose taken from the particle with the greatest associated
        probability. If multiple particles share the maximum probability, the
        goal pose will be randomly selected from those particles.

        Parameters
        ----------
        fusion_engine : FusionEngine
            A fusion engine with a particle filter.

        Returns
        -------
        array_like
            A pose as [x,y,theta] in [m,m,degrees].

        """
        target = self.robot.mission_planner.target
        fusion_engine = self.robot.fusion_engine
        # <>TODO: @Nick Test this!
        if fusion_engine.filter_type != 'particle':
                raise ValueError('The fusion_engine must have a '
                                 'particle_filter.')

        theta = random.uniform(0, 360)

        # If no target is specified, do default behavior
        if target is None:
            if len(fusion_engine.filters) > 1:
                particles = fusion_engine.filters['combined'].particles
            else:
                particles = next(fusion_engine.filters.iteritems()).particles
        else:
            try:
                particles = fusion_engine.filters[target].particles
                logging.info('Looking for {}'.format(target))
            except:
                logging.warn('No particle filter found for specified target')
                return None

        max_prob = particles[:, 0].max()
        max_particle_i = np.where(particles[:, 0] == max_prob)[0]
        max_particles = particles[max_particle_i, :]

        # Select randomly from max_particles
        max_particle = random.choice(max_particles)
        goal_pose = np.append(max_particle[1:3], theta)

        return goal_pose

    def update(self):

        super(ParticleGoalPlanner,self).update()
