#!/usr/bin/env python

""" The 'probability' goal planner subclass of GoalPlanner
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

class ProbabilityGoalPlanner(GoalPlanner):

	def __init__(self, robot, type_='stationary', view_distance=0.3,
                    use_target_as_goal=True, goal_pose_topic=None, **kwargs):

        super(ProbabilityGoalPlanner, self).__init__(robot=robot,
                                                type_=type_,
                                                view_distance=view_distance,
                                                use_target_as_goal=use_target_as_goal,
                                                goal_pose_topic=goal_pose_topic)

	 def find_goal_pose(self):
        """Find a goal pose from the point of highest probability (the
            Maximum A Posteriori, or MAP, point).

        Parameters
        ----------
        fusion_engine : FusionEngine
            A fusion engine with a probabilistic filter.

        Returns
        -------
        array_like
            A pose as [x,y,theta] in [m,m,degrees].

        """
        target = self.robot.mission_planner.target
        fusion_engine = self.robot.fusion_engine

        # <>TODO: @Nick Test this!
        if fusion_engine.filter_type != 'gauss sum':
                raise ValueError('The fusion_engine must have a '
                                 'gauss sum filter.')

        theta = random.uniform(0, 360)

        # If no target is specified, do default behavior
        if target is None:
            if len(fusion_engine.filters) > 1:
                posterior = fusion_engine.filters['combined'].probability
            else:
                posterior = next(fusion_engine.filters.iteritems()).probability
        else:
            try:
                posterior = fusion_engine.filters[target].probability
                logging.info('Looking for {}'.format(target))
            except:
                logging.warn('No gauss sum filter found for specified target')
                return None

        bounds = self.feasible_layer.bounds
        MAP_point, MAP_prob = posterior.max_point_by_grid(bounds)

        # Select randomly from max_particles
        goal_pose = np.append(MAP_point, theta)

        # Check if its a feasible goal
        pt = Point(goal_pose[0:2])
        if self.feasible_layer.pose_region.contains(pt):
            # If feasible return goal
            return goal_pose
        else:
            # Check if it inside an object, update probability
            map_ = self.robot.map
            for name, obj in map_.objects.iteritems():
                if obj.shape.contains(pt):
                    self.clear_probability_from_objects(target, obj)
                    goal_pose = None
                    return goal_pose
            else:
                # If not in feasible region but not in object, and not
                # using a view goal, generate a point in the same way
                # as a view goal
                logging.info('Not feasible, not in object')
                if self.use_target_as_goal:
                    goal_pose = self.view_goal(goal_pose)

        return goal_pose

    def clear_probability_from_objects(self, target, obj):
        logging.info('Clearing probability from {}'.format(obj.name))
        fusion_engine = self.robot.fusion_engine
        vb = VariationalBayes()

        if not hasattr(obj, 'relations'):
            obj.define_relations()

        if target is not None:
            prior = fusion_engine.filters[target].probability
            likelihood = obj.relations.binary_models['Inside']
            mu, sigma, beta = vb.update(measurement='Not Inside',
                                        likelihood=likelihood,
                                        prior=prior,
                                        use_LWIS=False,
                                        )
            gm = GaussianMixture(beta, mu, sigma)
            fusion_engine.filters[target].probability = gm
        else:
            # Update all but combined filters
            for name, filter_ in fusion_engine.filters.iteritems():
                if name == 'combined':
                    pass
                else:
                    prior = filter_.probability
                    likelihood = obj.relations.binary_models['Inside']
                    mu, sigma, beta = vb.update(measurement='Not Inside',
                                                likelihood=likelihood,
                                                prior=prior,
                                                use_LWIS=False,
                                                )
                    gm = GaussianMixture(beta, mu, sigma)
                    filter_.probability = gm
    def update(self):

        super(ProbabilityGoalPlanner,self).update()
