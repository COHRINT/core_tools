#!/usr/bin/env python

from observation_interface.srv import *
from observation_interface.msg import *
from std_msgs.msg import String, Float64
import rospy

def observation_client(msg):
    rospy.wait_for_service('observation_interface')
    try:
        pt = rospy.ServiceProxy('observation_interface',observation_server)
        print(msg)
        print('')
        res = pt(msg)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == '__main__':
    msg = ObservationRequest()
    msg.name = 'deckard'
    msg.weights = [1,1,1]
    msg.means = [2,2,2]
    msg.variances = [3,3,3]

    obs = observation_client(msg)

    print(obs)
