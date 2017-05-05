from __future__ import division
'''
************************************************************************************************************************************************************
File: PolicyGenerator.py
Written By: Luke Burks
December 2016

This is intended as a template for POMDP policy 
generators. Ideally all problem specific bits
will have been removed

Input: -n <problemName> -b <initialBeliefNumber> -a <alphaSaveNumber> -m <maxNumMixands> -f <finalNumMixands> -g <generateNewModels> -s <useSoftmaxModels>
Output: solve function
<problemName>Alphas<alphaSaveNumber>.npy, a file containing the policy found by the generator
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


class PolicyGenerator:

	def __init__(self,argv):
	

		#Initialize exit flag
		self.exitFlag = False; 
		self.b = None; 
		signal.signal(signal.SIGINT, self.signal_handler);



		#Set up default arguments
		problemName = ''; 
		belNum = '1'; 
		self.alphaNum = '1'; 
		generate = False; 
		self.finalMix = 100; 
		self.maxMix = 10; 
		self.iterations = 1000; 
		self.useSoft = False; 

		#Grab command line arguments for problem
		try:
		  opts, args = getopt.getopt(argv[1:],"hn:b:a:m:f:g:s:",["name=","belNum=","alSaveNum=",'maxMix=','finalMix=','gen=','softmax='])
		except getopt.GetoptError:
		  print 'PolicyGenerator.py -n <problemName> -b <initialBeliefNumber> -a <alphaSaveNumber> -m <maxNumMixands> -f <finalNumMixands> -g <generateNewModels> -s <useSoftmaxModels>'
		  sys.exit(2)
		for opt, arg in opts:
		  if opt == '-h':
		    print 'PolicyGenerator.py -n <problemName> -b <initialBeliefNumber> -a <alphaSaveNumber> -m <maxNumMixands> -f <finalNumMixands> -g <generateNewModels> -s <useSoftmaxModels>'
		    sys.exit()
		  elif opt in ("-n", "--name"):
		    problemName = arg
		  elif opt in ("-b", "--belNum"):
		    belNum = arg
		  elif opt in ("-a", "--alLoadNum"):
		  	self.alphaNum = arg
		  elif opt in ("-m","--maxMix"):
		  	self.maxMix = int(arg); 
		  elif opt in ("-f","--finalMix"):
		  	self.finalMix = int(arg); 
		  elif opt in ("-g","--gen"):
		  	if(arg == 'True'):
		  		generate = True; 
		  	else:
		  		generate = False;  
		  elif opt in ("-s","--softmax"):
		  	if(arg == 'True'):
		  		self.useSoft = True; 
		  	else:
		  		self.useSoft = False; 

		if(problemName == ''):
			print('Input Problem Name'); 
			problemName = raw_input(); 


		belLoad = '../beliefs/' + problemName + 'Beliefs' + belNum + '.npy'; 
		self.alSave = '../policies/' + problemName + 'Alphas' + self.alphaNum + '.npy'; 
		modelPath = '../models/'+ problemName + 'Model'; 
		modelName = problemName+'Model'; 

		self.problemName = problemName; 

		#import the specified model
		sys.path.append('../models/') 
		modelModule = __import__(modelName, globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;


		#Grab Modeling Code
		allMod = modelClass(); 

		#Build Transition Model
		allMod.buildTransition(); 
		self.delA = allMod.delA; 
		self.delAVar = allMod.delAVar; 

		#Build Observation Model
		if(generate==True):
			print("Building Observation Models"); 
		allMod.buildObs(gen=generate);
		self.pz = allMod.pz;
		
		print(self.pz); 

		#Build Reward Model
		if(generate == True):
			print("Building Reward Model"); 
		allMod.buildReward(gen=generate); 
		self.r = allMod.r; 
		self.discount = allMod.discount; 
	

		#Loading Beliefs
		if(belLoad is None):
			print("No belief file"); 
			sys.exit();
		try:
			self.B = np.load(belLoad).tolist(); 
		except:
			print('Belief file not found'); 
			raise; 
		
			
		#Initialize Gamma
		self.Gamma = copy.deepcopy(self.r); 
		for i in range(0,len(self.Gamma)):
			self.Gamma[i].action = i; 
			for g in self.Gamma[i].Gs:
				g.weight = g.weight/(1-self.discount);
			self.Gamma[i] = self.Gamma[i].kmeansCondensationN(k=self.finalMix);  
		
		
		 


	#Generate the approximate POMDP policy
	def solve(self,verbose=True):

		startTime = 0; 
		iterationTimes = []; 
		for counter in range(0,self.iterations):
			
			if(counter==0):
				iterationTimes = [startTime]; 
			else:
				iterationTimes.append(time.clock()-iterationTimes[counter-1]); 

			if(self.exitFlag):
				break; 

			if(verbose):
				print("Iteration: " + str(counter+1)); 
			else:
				print("Iteration: " + str(counter+1)); 
			
			bestAlphas = [GM()]*len(self.B); 
			Value = [0]*len(self.B); 

			for b in self.B:
				bestAlphas[self.B.index(b)] = self.Gamma[np.argmax([self.continuousDot(self.Gamma[j],b) for j in range(0,len(self.Gamma))])];
				Value[self.B.index(b)] = self.continuousDot(bestAlphas[self.B.index(b)],b); 
				
			GammaNew = [];

			BTilde = copy.deepcopy(self.B); 
			
			if(self.useSoft):
				self.preComputeAlsSoftmax(); 
			else:
				self.preComputeAls(); 

			while(len(BTilde) > 0):

				if(self.exitFlag):
					break; 

				b = random.choice(BTilde); 

				BTilde.remove(b); 

				al = self.backup(b); 

			
				if(self.continuousDot(al,b) < Value[self.findB(b)]):
					index = 0; 
					for h in self.B:
						if(b.fullComp(h)):
							index = self.B.index(h); 
					al = bestAlphas[index]; 
				else:
					index = 0; 
					for h in self.B:
						if(b.fullComp(h)):
							index = self.B.index(h);
					bestAlphas[index] = al; 

				#remove from Btilde all b for which this alpha is better than its current
				for bprime in BTilde:
					if(self.continuousDot(al,bprime) >= Value[self.findB(bprime)]):
						BTilde.remove(bprime); 



				#make sure the alpha doesn't already exist
				addFlag = True; 
				for i in range(0,len(GammaNew)):
					if(al.fullComp(GammaNew[i])):
						addFlag = False; 
				if(addFlag):
					GammaNew += [al];


			
			if(verbose and self.exitFlag == False):
				print("Number of Alphas: " + str(len(GammaNew))); 
				av = 0; 
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size; 
				av = av/len(GammaNew);  
				print("Average number of mixands: " + str(av)); 
			if(self.exitFlag == False):
				if(counter < self.iterations-1):
					for i in range(0,len(GammaNew)):
						#if(GammaNew[i].size > maxMix):
							#GammaNew[i].condense(max_num_mixands=self.maxMix);
						GammaNew[i] = GammaNew[i].kmeansCondensationN(k = self.maxMix); 
				elif(counter == self.iterations-1):
					for i in range(0,len(GammaNew)):
						#GammaNew[i].condense(max_num_mixands=self.finalMix);
						GammaNew[i] = GammaNew[i].kmeansCondensationN(k = self.finalMix); 

			if(verbose and self.exitFlag == False):
				#GammaNew[0].display(); 
				av = 0; 
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size; 
				av = av/len(GammaNew);  
				print("Reduced average number of mixands: " + str(av)); 
				print("Actions: " + str([GammaNew[i].action for i in range(0,len(GammaNew))])); 
				print("");


			if(self.exitFlag == False):
				f = open(self.alSave,"w"); 
				self.Gamma = copy.deepcopy(GammaNew); 
				np.save(f,self.Gamma); 
				f.close(); 
				
		if(not os.path.isdir('../results/'+self.problemName)):
			os.mkdir('../results/'+self.problemName); 
		f = open('../results/'+self.problemName+'/' + self.problemName + '_Timing' + self.alphaNum +  '.npy','w+'); 
		np.save(f,iterationTimes);
		f.close();

		f = open(self.alSave,"w+"); 
		np.save(f,self.Gamma); 
		f.close(); 


	#Compute Intermediate Alpha functions for each action, observation, and previous alpha
	def preComputeAls(self):
		G = self.Gamma; 
		R = self.r; 
		pz = self.pz; 



		als1 = [[[0 for i in range(0,len(pz))] for j in range(0,len(self.delA))] for k in range(0,len(G))]; 

		for j in range(0,len(G)):
			for a in range(0,len(self.delA)):
				for o in range(0,len(pz)):
					als1[j][a][o] = GM(); 
					for k in range(0,G[j].size):
						for l in range(0,pz[o].size): 
							#get weights wk,wl, and del

							weight = G[j].Gs[k].weight*pz[o].Gs[l].weight*mvn.pdf(pz[o].Gs[l].mean,G[j].Gs[k].mean,(np.matrix(G[j].Gs[k].var)+np.matrix(pz[o].Gs[l].var)).tolist()); 

							#get sig and ss
							sigtmp = (np.matrix(G[j].Gs[k].var).I + np.matrix(pz[o].Gs[l].var)).tolist(); 
							sig = np.matrix(sigtmp).I.tolist(); 
						
							sstmp = np.matrix(G[j].Gs[k].var).I*np.transpose(np.matrix(G[j].Gs[k].mean)) + np.matrix(pz[o].Gs[l].var).I*np.transpose(np.matrix(pz[o].Gs[l].mean)); 
							ss = np.dot(sig,sstmp).tolist(); 


							smean = (np.transpose(np.matrix(ss)) - np.matrix(self.delA[a])).tolist(); 
							sigvar = (np.matrix(sig)+np.matrix(self.delAVar)).tolist(); 
			
								
							als1[j][a][o].addG(Gaussian(smean[0],sigvar,weight)); 
		self.preAls = als1; 


	#Compute Intermediate Alpha functions for each action, observation, and previous alpha
	#Observations assumed to be softmax functions
	def preComputeAlsSoftmax(self):
		G = self.Gamma;  

		als1 = [[[0 for i in range(0,self.pz.size)] for j in range(0,len(self.delA))] for k in range(0,len(G))]; 

		for j in range(0,len(G)):
			for a in range(0,len(self.delA)):
				for o in range(0,self.pz.size):
					als1[j][a][o] = GM(); 

					#alObs = G[j].runVB(self.soft_weight,self.soft_bias,self.soft_alpha,self.soft_zeta_c,softClassNum = o); 
					alObs = self.pz.runVBND(G[j],o); 

					for k in alObs.Gs:
						mean = (np.matrix(k.mean) - np.matrix(self.delA[a])).tolist(); 
						var = (np.matrix(k.var) + np.matrix(self.delAVar)).tolist(); 
						weight = k.weight; 
						als1[j][a][o].addG(Gaussian(mean,var,weight)); 

		self.preAls = als1; 


	def backup(self,b):
		G = self.Gamma; 
		R = self.r; 
		pz = self.pz; 

		if(self.useSoft):
			obslen = pz.size; 
		else:
			obslen = len(pz); 

		als1 = self.preAls; 
		

		bestVal = -10000000000; 
		bestAct= 0; 
		bestGM = []; 

		for a in range(0,len(self.delA)):
			suma = GM(); 
			for o in range(0,obslen):
				suma.addGM(als1[np.argmax([self.continuousDot(als1[j][a][o],b) for j in range(0,len(als1))])][a][o]); 
			suma.scalerMultiply(self.discount); 
			suma.addGM(R[a]); 

			tmp = self.continuousDot(suma,b);
			#print(a,tmp); 
			if(tmp > bestVal):
				bestAct = a; 
				bestGM = copy.deepcopy(suma); 
				bestVal = tmp; 

		bestGM.action = bestAct; 

		return bestGM;  


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


	def signal_handler(self,signal, frame):
		print("Stopping Policiy Generation and printing to file"); 
		self.exitFlag = True; 




if __name__ == "__main__":

	a = PolicyGenerator(sys.argv); 
	a.solve(); 

	

	
