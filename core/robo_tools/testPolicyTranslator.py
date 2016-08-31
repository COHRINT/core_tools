#!/usr/bin/env python

"""
test policy translator for pomdp goal planning in
    cops and robots experiment
"""

import math

class testPolicyTranslator(object):

    def __init__(self):
        self.possible_actions = ['left','right','up','down']

    def getAction(self,current_position):
        if current_position[0] > 2:# and current_position[1] < 2:
            action = 'left'
            return action
        elif current_position[0] < 2:# and current_position[1] > 2:
            action = 'right'
            return action
        # elif current_position[0] < 2 and current_position < 2:
        #     action = 'up'
        #     return action
        # elif current_position[0] > 2 and current_position[1] > 2:
        #     action = 'down'
        #     return action

    def getNextPose(self,current_position):
        action = self.getAction(current_position)
        if action == 'left':
            orientation = math.pi/2
            goal_pose = [current_position[0]-2,
                            current_position[1],
                            orientation]
            return goal_pose
        elif action == 'right':
            orientation = math.pi/2
            goal_pose = [current_position[0]+2,
                            current_position[1],
                            orientation]
            return goal_pose
        elif action == 'up':
            orientation = math.pi/2
            goal_pose = [current_position[0],
                            current_position[1]+2,
                            orientation]
            return goal_pose
        elif action == 'down':
            orientation = -(math.pi/2)
            goal_pose = [current_position[0],
                            current_position[1]-2,
                            orientation]
            return goal_pose
