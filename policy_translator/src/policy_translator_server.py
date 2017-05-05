#!/usr/bin/env python

from policy_translator.srv import *
from policy_translator.msg import *
from observation_interface.msg import *
from observation_interface.srv import *

import rospy
import tf
import numpy as np
import math

from gaussianMixtures import GM
from PolicyTranslator import PolicyTranslator
from belief_handling import rehydrate_msg, dehydrate_msg



#<>TODO: finish translator_wrapper function
# <>TODO: determine if method to instantiate translator object works

class PolicyTranslatorServer(object):

    def __init__(self):

        args = ['PolicyTranslator.py','-n','D2Diffs','-r','True','-a','1','-s','False','-g','True'];
        self.pt = PolicyTranslator(args)

        rospy.init_node('policy_translator_server')
        self.listener = tf.TransformListener()
        s = rospy.Service('translator',policy_translator_service,self.handle_policy_translator)
        print('Policy translator service ready.')
        rospy.spin()

    def handle_policy_translator(self,req):
        name = req.request.name

        if not req.request.weights:
            obs = None
        else:
            obs_msg = ObservationRequest()
            obs = self.obs_server_client(obs_msg)

        belief = self.translator_wrapper(req.request.name,req.request.weights,
                                    req.request.means,req.request.variances,obs)

        # print('*********')
        weights_updated = belief[0]
        # print(weights_updated)
        means_updated = belief[1]
        # print(means_updated)
        variances_updated = belief[2]
        # print(variances_updated)
        goal_pose = belief[3]

        res = self.create_message(req.request.name,
                            weights_updated,
                            means_updated,
                            variances_updated,
                            goal_pose)

        return res

    def obs_server_client(self,msg):
        rospy.wait_for_service('observation_interface')
        try:
            proxy = rospy.ServiceProxy('observation_interface',observation_server)
            res = proxy(msg)
            return res.response.observation
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def tf_update(self,name):
        name = name.lower()
        ref = "/" + name + "/odom"
        child = "/" + name + "/base_footprint"
        (trans, rot) = self.listener.lookupTransform(ref, child, rospy.Time(0))
        x = trans[0]
        y = trans[1]
        (_, _, theta) = tf.transformations.euler_from_quaternion(rot)
        pose = [x, y, np.rad2deg(theta)]
        return pose

    def translator_wrapper(self,name,weights,means,variances,obs):
        '''
        LUKES CODE WILL BE CALLED IN THIS FUNCTION
        '''

        belief = rehydrate_msg(weights,means,variances)

        position = self.tf_update(name)

        (b_updated,goal_pose) = self.pt.getNextPose(belief,obs,[position[0],position[1]])

        if b_updated is not None:
            (weights,means,variances) = dehydrate_msg(b_updated)
            # print(variances)

        orientation = math.atan2(goal_pose[1]-position[1],goal_pose[0]-position[0])
        goal_pose.append(orientation)



        belief = [weights,means,variances,goal_pose]

        # print('@@@@@@@@@')
        # print(weights)


        return belief

    def create_message(self,name,weights,means,variances,goal_pose):
        msg = PolicyTranslatorResponse()
        msg.name = name
        # print('&&&&&&&')
        # print msg.name
        msg.weights_updated = weights
        # print msg.weights_updated
        msg.means_updated = means
        # print msg.means_updated
        msg.variances_updated = variances
        # print msg.variances_updated
        msg.goal_pose = goal_pose
        # print goal_pose
        return msg

if __name__ == "__main__":
    PolicyTranslatorServer()
