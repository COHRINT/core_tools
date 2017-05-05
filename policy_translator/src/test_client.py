#!/usr/bin/env python

from test_service.srv import *
from test_service.msg import *
from std_msgs.msg import String, Float64
import rospy

def policy_translator_client(msg):
    rospy.wait_for_service('translator')
    try:
        pt = rospy.ServiceProxy('translator',policy_translator)
        print(msg)
        print('')
        res = pt(msg)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def dehydrate_msg(belief,n):
    '''
    INSERT CODE TO GET WEIGHTS MEANS VARIANCES FROM BELIEF HERE
    '''
    means = belief
    # Means - list of 1xn vectors
    means_length = len(means)
    total_elements = n*means_length
    means_dry = []

    for i in range(0,means_length):
        for j in range(0,n):
            means_dry.append(means[i][j])

    return means_dry

def rehydrate_msg(res):
    weights = res.response.weights_updated
    means = res.response.means_updated
    variances = res.response.variances_updated
    size = res.response.size



if __name__ == '__main__':
    msg = PolicyTranslatorRequest()
    # name = String('deckard')
    msg.name = 'deckard'
    msg.weights = [1.1,2.2,3.3,4.4,5.5]
    msg.means = [1.1,2.2,3.3,4.4,5.5]
    msg.variances = [1.1,2.2,3.3,4.4,5.5]

    print('Requesting service')
    res = policy_translator_client(msg)
    print(res)

    for i in range(0,5):

        msg = PolicyTranslatorRequest()
        msg.name = 'deckard'
        msg.weights = res.response.weights_updated
        msg.means = res.response.means_updated
        msg.variances = res.response.variances_updated

        print('Requesting service')
        res = policy_translator_client(msg)
        print(res)
        print('--------------------------')
        rospy.sleep(0.5)

    # mean = [1,2,3,4]
    # means = [mean,mean,mean,mean]
    #
    # variance1 = []
    #
    # means_dry = dehydrate_msg(means,len(mean))
    #
    # print(means_dry)
