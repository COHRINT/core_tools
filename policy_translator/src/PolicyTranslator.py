
from __future__ import division
'''
************************************************************************************************************************************************************
File: PolicyTranslator.py
Written By: Luke Burks
December 2016

This is intended as a template for POMDP policy
translators. Ideally all problem specific bits
will have been removed

Input: -n <problemName> -b <beliefSaveNumber> -a <alphaLoadNumber> -m <maxNumMixands> -g <greedySim> -s <useSoftmaxModels> -t <simType>
Output: simulation data

************************************************************************************************************************************************************
'''



__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"


import numpy as np
from scipy.stats import multivariate_normal as mvn
import random
import copy
import cProfile
import re
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import os;
from math import sqrt
import signal
import sys, getopt
import cProfile
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
import matplotlib.animation as animation
from numpy import arange
import time
import matplotlib.image as mgimg



class PolicyTranslator:

	def __init__(self, argv):
		#Initialize exit flag
		self.exitFlag = False;
		self.b = None;
		signal.signal(signal.SIGINT, self.signal_handler);


		#Set up default arguments
		self.problemName = '';
		self.belNum = '1';
		self.alphaNum = '1';
		generate = False;
		self.maxMix = 10;
		self.iterations = 1000;
		self.useSoft = False;
		simNum = -1;
		self.greedy = False;
		robots = False;

		#Grab command line arguments for problem
		try:
		  opts, args = getopt.getopt(argv[1:],"hn:b:a:m:f:g:s:t:r:",["name=","belNum=","alLoadNum=",'maxMix=','greed=','softmax=','simType=','robots='])
		except getopt.GetoptError:
		  print 'PolicyTranslator.py -n <problemName> -b <beliefSaveNumber> -a <alphaLoadNumber> -m <maxNumMixands> -g <greedySim> -s <useSoftmaxModels> -t <simType> -r <robots>'
		  sys.exit(2)
		for opt, arg in opts:
		  if opt == '-h':
		    print 'PolicyTranslator.py -n <problemName> -b <beliefSaveNumber> -a <alphaLoadNumber> -m <maxNumMixands> -g <greedySim> -s <useSoftmaxModels> -t <simType> -r <robots>'
		    sys.exit()
		  elif opt in ("-n", "--name"):
		    self.problemName = arg
		  elif opt in ("-b", "--belNum"):
		    self.belNum = arg
		  elif opt in ("-a", "--alLoadNum"):
		  	self.alphaNum = arg
		  elif opt in ("-m","--maxMix"):
		  	self.maxMix = int(arg);
		  elif opt in ("-g","--greed"):
		  	if(arg == 'True'):
		  		self.greedy = True;
		  	else:
		  		self.greedy = False;
		  elif opt in ("-s","--softmax"):
		  	if(arg == 'True'):
		  		self.useSoft = True;
		  	else:
		  		self.useSoft = False;
		  elif opt in ("-t","--simType"):
		  	if(arg == '0'):
		  		simNum = 0;
		  	else:
		  		simNum = int(arg);
		  elif opt in ("-r","--robots"):
		  	if(arg == 'True'):
		  		robots = True;


		if(self.problemName == ''):
			print('Input Problem Name');
			self.problemName = raw_input();


		self.belSave = os.path.dirname(__file__) + '/' + '../beliefs/' + self.problemName + 'Beliefs' + self.belNum + '.npy';
		alLoad = os.path.dirname(__file__) + '/' + '../policies/' + self.problemName + 'Alphas' + self.alphaNum + '.npy';
		modelPath = os.path.dirname(__file__) + '/' + '../models/'+ self.problemName + 'Model';
		modelName = self.problemName+'Model';




		#import the specified model
		sys.path.append(os.path.dirname(__file__) + '/../models/')
		modelModule = __import__(modelName, globals(), locals(), ['ModelSpec'],0);
		modelClass = modelModule.ModelSpec;


		#Grab Modeling Code
		allMod = modelClass();


		#Build Transition Model
		allMod.buildTransition();
		self.delA = allMod.delA;
		self.delAVar = allMod.delAVar;
		self.bounds = allMod.bounds;


		#Build Observation Model
		if(generate==True):
			print("Building Observation Models");
		allMod.buildObs(gen=generate);
		self.pz = allMod.pz;

		#Build Reward Model
		if(generate == True):
			print("Building Reward Model");
		allMod.buildReward(gen=generate);
		self.r = allMod.r;
		self.bounds = allMod.bounds;


		#Initialize Gamma
		self.loadPolicy(alLoad);


		if(simNum == 0 and not robots):
			self.generateBeliefs();
		elif(simNum == 1 and not robots):
			self.runSingleSim(greedySim = self.greedy);
		elif(not robots):
			self.runMultiSim(simCount = simNum,greedySim = self.greedy);



	def loadPolicy(self,fileName):
		self.Gamma = np.load(fileName);

	def getAction(self,b):
		act = self.Gamma[np.argmax([self.continuousDot(j,b) for j in self.Gamma])].action;
		return act;


	def getGreedyAction(self,bel):

		MAP = bel.findMAPN();

		if(abs(MAP[0])>abs(MAP[1])):
			if(MAP[0] > 0):
				act = 0;
			else:
				act = 1;
		else:
			if(MAP[1] > 0):
				act = 2;
			else:
				act = 3;

		return act;


	def covAdd(self,a,b):
		if(type(b) is not list):
			b = b.tolist();
		if(type(a) is not list):
			a = a.tolist();

		c = copy.deepcopy(a);

		for i in range(0,len(a)):
			for j in range(0,len(a[i])):
				c[i][j] += b[i][j];
		return c;



	def findB(self,b):
		for beta in self.B:
			if(beta.fullComp(b)):
				return self.B.index(beta);


	def continuousDot(self,a,b):
		suma = 0;

		if(isinstance(a,np.ndarray)):
			a = a.tolist();
			a = a[0];

		if(isinstance(a,list)):
			a = a[0];

		a.clean();
		b.clean();

		for k in range(0,a.size):
			for l in range(0,b.size):
				suma += a.Gs[k].weight*b.Gs[l].weight*mvn.pdf(b.Gs[l].mean,a.Gs[k].mean, np.matrix(a.Gs[k].var)+np.matrix(b.Gs[l].var));
		return suma;


	def beliefUpdate(self,b,a,o):
		btmp = GM();

		for obs in self.pz[o].Gs:
			for bel in b.Gs:
				sj = np.matrix(bel.mean).T;
				si = np.matrix(obs.mean).T;
				delA = np.matrix(self.delA[a]).T;
				sigi = np.matrix(obs.var);
				sigj = np.matrix(bel.var);
				delAVar = np.matrix(self.delAVar);

				weight = obs.weight*bel.weight;
				weight = weight*mvn.pdf((sj+delA).T.tolist()[0],si.T.tolist()[0],np.add(sigi,sigj,delAVar));
				var = (sigi.I + (sigj+delAVar).I).I;
				mean = var*(sigi.I*si + (sigj+delAVar).I*(sj+delA));
				weight = weight.tolist();
				mean = mean.T.tolist()[0];
				var = var.tolist();


				btmp.addG(Gaussian(mean,var,weight));
		btmp.normalizeWeights();
		btmp = btmp.kmeansCondensationN(self.maxMix);
		#btmp.condense(maxMix);
		btmp.normalizeWeights();
		return btmp;


	def beliefUpdateSoftmax(self,b,a,o):

		btmp = GM();
		btmp1 = GM();
		for j in b.Gs:
			mean = (np.matrix(j.mean) + np.matrix(self.delA[a])).tolist()[0];
			var = (np.matrix(j.var) + np.matrix(self.delAVar)).tolist();
			weight = j.weight;
			btmp1.addG(Gaussian(mean,var,weight));
		btmp = self.pz.runVBND(btmp1,o);

		#btmp.condense(maxMix);
		btmp = btmp.kmeansCondensationN(self.maxMix);
		btmp.normalizeWeights();

		return btmp;


	def distance(self,x1,y1,x2,y2):
		dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
		dist = math.sqrt(dist);
		return dist;

	def generateBeliefs(self,numSims = 10, numStep = 10):

		allSimB = [];
		for i in range(0,numSims):
			#Randomly choose initial state
			inPose = [0 for i in range(0,len(self.delA[0]))];
			for j in range(0,len(self.delA[0])):
				inPose[j] = random.random()*self.bounds[j][1] + self.bounds[j][0];
			#Simulate
			[allB,allX,allXInd,allAct,allReward] = self.simulate(initialPose = inPose,numSteps = numStep,belGen = True);
			#Store
			allSimB.append(allB);
		#Save
		f = open(self.belSave,"w");
		np.save(f,allSimB);

	def runMultiSim(self,simCount=10,simSteps = 100,greedySim = False):
		#run simulations
		allSimRewards = [];
		allSimAct = [];
		allSimX = [];
		allSimXInd = [];
		allSimB = [];
		for i in range(0,simCount):
			inPose = [0 for j in range(0,len(self.delA[0]))];
			for j in range(0,len(self.delA[0])):
				inPose[j] = random.random()*(self.bounds[j][1]-self.bounds[j][0]) + self.bounds[j][0];


			print("Starting simulation: " + str(i+1) + " of " + str(simCount) + " with initial position: " + str(inPose));
			[allB,allX,allXInd,allAct,allReward] = self.simulate(initialPose = inPose,numSteps = simSteps,greedy=greedySim);

			allSimRewards.append(allReward);
			allSimB.append([allB]);
			allSimAct.append([allAct]);
			allSimX.append([allX]);
			allSimXInd.append([allXInd]);
			print("Simulation complete. Reward: " + str(allReward[i-1]));


		#save all data
		dataSave = {"Beliefs":allSimB,"States":allSimX,"States(Ind)":allSimXInd,"Actions":allSimAct,'Rewards':allSimRewards};

		if(not os.path.isdir('../results/'+self.problemName)):
			os.mkdir('../results/'+self.problemName);
		if(not greedySim):
			f = open('../results/'+self.problemName+'/' + self.problemName + '_Data' + self.alphaNum +  '.npy','w+');
		else:
			f = open('../results/'+self.problemName+'/' + self.problemName + '_Data_Greedy' + self.alphaNum +  '.npy','w+');
		np.save(f,dataSave);
		f.close();

	def runSingleSim(self,simSteps = 100,greedySim = False):
		#Run a simulation
		#Randomly choose initial state
		inPose = [0 for i in range(0,len(self.delA[0]))];
		for j in range(0,len(self.delA[0])):
			inPose[j] = random.random()*self.bounds[j][1] + self.bounds[j][0];

		#run simulation
		[allB,allX,allXInd,allAct,allReward] = self.simulate(initialPose = inPose,numSteps = simSteps,greedy=greedySim);

		#Show Results
		fig,ax = plt.subplots(1,sharex=True);

		for i in range(0,len(allB)):
			[x,y,c] = allB[i].plot2D(low = [self.bounds[0][0],self.bounds[1][0]],high = [self.bounds[0][1],self.bounds[1][1]],vis = False);
			ax.cla()
			ax.contourf(x,y,c,cmap ='viridis');
			ax.set_title('time step='+str(i))
			ax.set_xlabel('position (m)')
			ax.set_ylabel('position (m)')
			ax.scatter(allX[i][0],allX[i][1],c='r');

			#grab temp images
			fig.savefig('../tmp/img'+str(i)+".png",bbox_inches='tight',pad_inches=0)
			plt.pause(.1)


		#Animate Results
		fig,ax=plt.subplots()
		images=[]
		for k in range(0,simSteps):
			fname='../tmp/img%d.png' %k
			img=mgimg.imread(fname)
			imgplot=plt.imshow(img)
			plt.axis('off')
			images.append([imgplot])
		ani=animation.ArtistAnimation(fig,images,interval=20)

		if(not os.path.isdir('../results/'+self.problemName)):
			os.mkdir('../results/'+self.problemName);
		if(not greedySim):
			path = '../results/'+self.problemName+'/' + self.problemName + '_Animation' + self.alphaNum +  '.gif';
		else:
			path='../results/'+self.problemName+'/' + self.problemName + '_Animation_Greedy' + self.alphaNum + '.gif';

		ani.save(path,fps=2,writer='animation.writer')

	def simulate(self,initialPose = [1,4],initialBelief = None, numSteps = 20,greedy = False,belGen = False):

		#load initial belief
		if(initialBelief == None):
			b = GM();
			mean = [0]*len(self.delA[0]);
			var = [[0 for k in range(0,len(self.delA[0]))] for j in range(0,len(self.delA[0]))];
			for k in range(0,len(self.delA[0])):
				mean[k] = random.random()*(self.bounds[k][1]-self.bounds[k][0]) + self.bounds[k][0];
				var[k][k] = random.random()*10;
			b.addG(Gaussian(mean,var,1));
		else:
			b = initialBelief;

		#Setup data gathering
		x = initialPose;
		allX = [];
		allX.append(x);
		allXInd = [0]*len(self.delA[0]);
		for i in range(0,len(self.delA[0])):
			allXInd[i] = [x[i]];

		reward = 0;
		allReward = [0];
		allB = [];
		allB.append(b);

		allAct = [];


		#Simulate
		for count in range(0,numSteps):
			if(self.exitFlag):
				break;

			#Get action
			if(greedy):
				act = self.getGreedyAction(b);
			elif(belGen):
				act = random.randint(0,len(self.delA)-1);
			else:
				act = self.getAction(b);

			#Take action
			x = np.random.multivariate_normal(np.array(x)+np.array(self.delA[act]),self.delAVar,size =1)[0].tolist();

			#bound the movement
			for i in range(0,len(x)):
				x[i] = max(self.bounds[i][0],x[i]);
				x[i] = min(self.bounds[i][1],x[i]);

			#Get observation and update belief
			if(not self.useSoft):
				ztrial = [0]*len(self.pz);
				for i in range(0,len(self.pz)):
					ztrial[i] = self.pz[i].pointEval(x);
				z = ztrial.index(max(ztrial));
				b = self.beliefUpdate(b,act,z);
			else:
				ztrial = [0]*self.pz.size;
				for i in range(0,self.pz.size):
					ztrial[i] = self.pz.pointEval2D(i,x);
				z = ztrial.index(max(ztrial));
				b = self.beliefUpdateSoftmax(b,act,z);

			#save data
			allB.append(b);
			allX.append(x);
			allAct.append(act);
			for i in range(0,len(x)):
				allXInd[i].append(x[i]);

			reward += self.r[act].pointEval(x);
			allReward.append(reward);


		allAct.append(-1);

		#print("Simulation Complete. Accumulated Reward: " + str(reward));
		return [allB,allX,allXInd,allAct,allReward];


	def getNextPose(self,b,o,x):

		z = 0;
		if(o == 4):
			z = 0;
		elif(o==6):
			z = 1;
		elif(o==8):
			z = 2;
		elif(o==2):
			z = 3;
		elif(o==5):
			z = 4;

		x[0] = x[0]*2;
		x[1] = x[1]*2;

		copFlag = False;
		if(b is not None):
			copFlag = True;

		xorig = copy.deepcopy(x);

		if(not copFlag):
			act = 4;
			x = np.random.multivariate_normal(np.array(x)+np.array(self.delA[act]),self.delAVar,size =1)[0].tolist();
		else:
			#shift belief by cop position
			btilde = copy.deepcopy(b);
			for g in btilde:
				g.mean = (np.array(g.mean)*2-np.array(xorig)).tolist();

			copFlag = True;
			if(self.greedy):
				print('greed');
				act = self.getGreedyAction(btilde);
				if(act == 3):
					act = 2;
				elif(act == 2):
					act = 3;

			else:
				act = self.getAction(btilde);
			act = z


			x = (np.array(x) - np.array(self.delA[act])).tolist()

		#update belief
		if(self.useSoft and copFlag):
			print('soft');
			b = self.beliefUpdateSoftmax(btilde,act,z)
		elif(copFlag):
			b = self.beliefUpdate(btilde,act,z);

		x[0] = x[0]/2;
		x[1] = x[1]/2;

		x[0] = min(x[0],5);
		x[0] = max(x[0],0);
		x[1] = min(x[1],5);
		x[1] = max(x[1],0);

		#send back belief and position
		if(copFlag):
			#shift back
			btilde = copy.deepcopy(b);
			for g in btilde:
				g.mean = (np.array(xorig)-np.array(g.mean)).tolist();
				g.mean[0] = g.mean[0]/2;
				g.mean[1] = g.mean[1]/2;
			return [btilde,x];
		else:
			return [b,x];


	def signal_handler(self,signal, frame):
		print("Stopping Simulation...");
		self.exitFlag = True;

def testGetNextPose():
	args = ['PolicyTranslator.py','-n','D2Diffs','-r','True','-a','1','-g','True'];
	a = PolicyTranslator(args);

	b = GM();
	b.addG(Gaussian([4,2],[[1,0],[0,1]],1));
	x = [2,7];
	obs = [2,2,2,2,2,2,2,0,0,0,0,0];

	for i in range(0,len(obs)):
		[b,x] = a.getNextPose(b,obs[i],x);
		print(b.findMAPN());
		x[0] = max(0,x[0]);
		x[0] = min(10,x[0]);
		x[1] = max(0,x[1]);
		x[1] = min(10,x[1]);
		'''
		print(x);
		[xx,yy,cc] = b.plot2D(low = [0,0],high=[10,10],vis = False);
		plt.contourf(xx,yy,cc,cmap='viridis');
		plt.scatter(x[0],x[1]);
		plt.pause(0.5);
		'''


if __name__ == "__main__":

	#a = PolicyTranslator(sys.argv);

	testGetNextPose();
