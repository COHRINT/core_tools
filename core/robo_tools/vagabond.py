#!/usr/bin/env python

#import logging

__author__ = ["Ian Loefgren", "Sierra Williams"]
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "2.0.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

from core.robo_tools.robot import Robot
from core.robo_tools.planner import MissionPlanner

class Vagabond(Robot):
    """The Vagabond subclass of the generic Robot type.
    Parameters
    ----------
    name : str
        The robot's name.
    **kwargs
        Arguments passed to the ``Vagabond`` superclass.
    Attributes
    ----------
    mission_statuses : {'on the run'}
    """
    mission_planner_defaults = {}
    goal_planner_defaults = {'type_': 'simple',
                             'use_target_as_goal': True}
    path_planner_defaults = {'type_': 'direct'}

    def __init__(self,
                 name,
                 pose=None,
                 pose_source='python',
                 map_cfg={},
                 mission_planner_cfg={},
                 goal_planner_cfg={},
                 path_planner_cfg={},
                 **kwargs):
        # Use class defaults for kwargs not included
        mp_cfg = Vagabond.mission_planner_defaults.copy()
        mp_cfg.update(mission_planner_cfg)
        gp_cfg = Vagabond.goal_planner_defaults.copy()
        gp_cfg.update(goal_planner_cfg)
        pp_cfg = Vagabond.path_planner_defaults.copy()
        pp_cfg.update(path_planner_cfg)
        super(Vagabond, self).__init__(name,
                                     pose=pose,
                                     pose_source=pose_source,
                                     goal_planner_cfg=gp_cfg,
                                     path_planner_cfg=pp_cfg,
                                     map_cfg=map_cfg,
                                     create_mission_planner=False,
                                     color_str='red',
                                     **kwargs)

        self.found_vagabond = {}
        self.mission_planner = VagabondMissionPlanner(self, **mp_cfg)

    def update(self,i=0):

        super(Vagabond,self).update(i)


class VagabondMissionPlanner(MissionPlanner):
    # """The Cop subclass of the generic MissionPlanner
    # """
    # mission_statuses = ['on the run', 'captured']

    def __init__(self, robot, mission_status='on the run'):

        super(VagabondMissionPlanner, self).__init__(robot,
                                                   mission_status=mission_status)

    def update(self):
        """Update the robot's status
        """
        pass
        # Does not make sence anymore but still needs the update
        # if self.robot.name in self.robot.found_vagabond.keys():
        #     self.mission_status = 'on the run'
        # if self.mission_status is 'on the run':
        #     self.stop_all_movement()
