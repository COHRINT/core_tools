from __future__ import division
'''
****************************************************
File: InterceptPolicyGenerator.py
Written By: Luke Burks
October 2016

This is just a simplified copy of the Intercept Policy
Generator for a full regular pomdp case.

Well now we're moving into a 4 dimensional case

****************************************************
'''



__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "0.1"
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
import sys
import cProfile
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
import matplotlib.animation as animation
from numpy import arange
import time
import matplotlib.image as mgimg

#TODO: Include stop-flag criteria


class InterceptTestGenerator:

	def __init__(self,beliefFile = None,dis = 0.9,gen=False,altObs = True,qGen = True,humObs = True):
		if(humObs):
			fig,ax = plt.subplots();
			self.axes = ax;

		self.humanObs = humObs;

		#Initialize exit flag
		self.exitFlag = False;
		self.b = None;
		self.buildTransition();
		if(gen==True):
			print("Building Observation Models");
		if(altObs):
			self.buildAltObs(gen=gen);
		else:
			self.buildObs(gen=gen);


		if(gen == True):
			print("Building Reward Model");
		self.buildReward(gen = gen);
		self.discount = dis;

		if(qGen == True):
			self.MDPValueIteration(False);
			self.solveQ();

		if(beliefFile == None):
			self.B = [0]*5;
			self.B[0] = GM();
			var = np.matrix([[1,0],[0,1]]);
			self.B[0].addG(Gaussian([2.5,2.5],var,1));

			self.B[1] = GM();
			self.B[1].addG(Gaussian([1,5],var,1));


			self.B[2] = GM();
			self.B[2].addG(Gaussian([5,1],var,1));

			self.B[3] = GM();
			self.B[3].addG(Gaussian([0,0],var,1));

			self.B[4] = GM();
			self.B[4].addG(Gaussian([5,5],var,1));



			for i in range(0,100):
				tmp = GM();
				tmp.addG(Gaussian([random.random()*5,random.random()*5],var,1));
				self.B.append(tmp);


		else:
			self.B = np.load(beliefFile).tolist();





		#Initialize Gamma
		self.Gamma = [copy.deepcopy(self.r)];
		#self.Gamma = [copy.deepcopy(self.r),copy.deepcopy(self.r),copy.deepcopy(self.r)];



		'''
		for i in range(0,3):
			self.Gamma[i].addG(Gaussian([0,0],[[100,0],[0,100]],-5));
			self.Gamma[i].action = i;
		'''

		#TODO: This stuff....
		for i in range(0,len(self.Gamma)):
			for j in range(0,len(self.Gamma[i].Gs)):
				self.Gamma[i].Gs[j].weight = -100000;
				#tmp = 0;








	def solve(self,N,maxMix = 20, finalMix = 50, verbose = False, alsave = "interceptAlphasTemp.npy"):

		for counter in range(0,N):

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


			self.preComputeAls();
			#self.newPreComputeAls();


			while(len(BTilde) > 0):

				if(self.exitFlag):
					break;

				b = random.choice(BTilde);

				BTilde.remove(b);

				al = self.backup(b);


				#TODO: You added the else here
				if(self.continuousDot(al,b) < Value[self.findB(b)]):
					index = 0;
					for h in self.B:
						if(b.comp(h)):
							index = self.B.index(h);
					al = bestAlphas[index];
				else:
					index = 0;
					for h in self.B:
						if(b.comp(h)):
							index = self.B.index(h);
					bestAlphas[index] = al;

				#remove from Btilde all b for which this alpha is better than its current
				for bprime in BTilde:
					if(self.continuousDot(al,bprime) >= Value[self.findB(bprime)]):
						BTilde.remove(bprime);

				GammaNew += [al];




			if(verbose and self.exitFlag == False):
				print("Number of Alphas: " + str(len(GammaNew)));
				av = 0;
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size;
				av = av/len(GammaNew);
				print("Average number of mixands: " + str(av));
			if(self.exitFlag == False):
				if(counter < N-1):
					for i in range(0,len(GammaNew)):
						#TODO: Switch back to kmeans
						#GammaNew[i].condense(max_num_mixands=maxMix);
						GammaNew[i] = GammaNew[i].kmeansCondensationN(k = maxMix);
				elif(counter == N-1):
					for i in range(0,len(GammaNew)):
						#GammaNew[i].condense(max_num_mixands=finalMix);
						GammaNew[i] = GammaNew[i].kmeansCondensationN(k = finalMix);

			if(verbose and self.exitFlag == False):
				#GammaNew[0].display();
				av = 0;
				for i in range(0,len(GammaNew)):
					av += GammaNew[i].size;
				av = av/len(GammaNew);
				print("Reduced number of mixands: " + str(av));
				print("Actions: " + str([GammaNew[i].action for i in range(0,len(GammaNew))]));
				print("");

			if(self.exitFlag == False):
				f = open(alsave,"w");
				np.save(f,self.Gamma);
				f.close();
				self.Gamma = copy.deepcopy(GammaNew);

			'''
			if((counter+1)%5 == 0):
				for i in range(0,len(self.Gamma)):
					fig1 = plt.figure();
					print(self.Gamma[i].action);
					self.Gamma[i].plot2D();
				for j in range(0,3):
					print(self.getAction(self.B[j]));
			'''


		f = open(alsave,"w");
		np.save(f,self.Gamma);
		f.close();



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


							smean = (np.transpose(np.matrix(ss)) + np.matrix(self.delA[a])).tolist();
							sigvar = (np.matrix(sig)+np.matrix(self.delAVar)).tolist();


							als1[j][a][o].addG(Gaussian(smean[0],sigvar,weight));
		self.preAls = als1;


	#based on the idea that 1-detect = not detect
	#only to be used for binary observations
	def newPreComputeAls(self):
		G = self.Gamma;
		R = self.r;
		pz = self.pz;

		als1 = [[[0 for i in range(0,len(pz))] for j in range(0,len(self.delA))] for k in range(0,len(G))];

		for j in range(0,len(G)):
			for a in range(0,len(self.delA)):
				o = 0;
				als1[j][a][o] = GM();
				for k in range(0,G[j].size):
					for l in range(0,pz[o].size):
						#get weights wk,wl, and del
						weight = G[j].Gs[k].weight*pz[o].Gs[l].weight*mvn.pdf(pz[o].Gs[l].mean,G[j].Gs[k].mean, self.covAdd(G[j].Gs[k].var,pz[o].Gs[l].var));

						#get sig and ss
						sig= (np.matrix(G[j].Gs[k].var).I + np.matrix(pz[o].Gs[l].var).I).I.tolist();


						sstmp = np.matrix(G[j].Gs[k].var).I*np.transpose(np.matrix(G[j].Gs[k].mean)) + np.matrix(pz[o].Gs[l].var).I*np.transpose(np.matrix(pz[o].Gs[l].mean));
						ss = np.dot(sig,sstmp);


						smean = (np.transpose(np.matrix(ss)) + np.matrix(self.delA[a])).tolist();
						sigvar = (np.matrix(sig)+np.matrix(self.delAVar)).tolist();


						als1[j][a][o].addG(Gaussian(smean[0],sigvar,weight));

				als1[j][a][1] = GM();
				o = 1;

				for k in range(0,G[j].size):
					kap = G[j].Gs[k];
					mean = (np.matrix(kap.mean) - np.matrix(self.delA[a])).tolist();
					var = (np.matrix(kap.var) + np.matrix(self.delAVar)).tolist();
					als1[j][a][o].addG(Gaussian(mean,var,kap.weight));

					for l in range(0,pz[0].size):

						op = pz[0].Gs[l];
						var = (np.matrix(kap.var) + np.matrix(op.var)).tolist();
						weight = kap.weight*op.weight*mvn.pdf(kap.mean,op.mean,var);




						c2 = (np.matrix(kap.var).I + np.matrix(op.var).I).I;
						c1 = c2*(np.matrix(kap.var).I*np.transpose(np.matrix(kap.mean)) + np.matrix(op.var).I*np.transpose(np.matrix(op.mean)))

						me = np.transpose((c1 - np.transpose(np.matrix(self.delA[a])))).tolist()[0];



						als1[j][a][o].addG(Gaussian(me,(c2+np.matrix(self.delAVar)).tolist(),-weight));





		self.preAls = als1;


	def backup(self,b):
		G = self.Gamma;
		R = self.r;
		pz = self.pz;

		als1 = self.preAls;


		#one alpha for each belief, so one per backup


		bestVal = -10000000000;
		bestAct= 0;
		bestGM = [];



		for a in range(0,len(self.delA)):
			suma = GM();
			for o in range(0,len(pz)):
				suma.addGM(als1[np.argmax([self.continuousDot(als1[j][a][o],b) for j in range(0,len(als1))])][a][o]);
			suma.scalerMultiply(self.discount);
			suma.addGM(R);

			tmp = self.continuousDot(suma,b);
			#print(a,tmp);
			if(tmp > bestVal):
				bestAct = a;
				bestGM = suma;
				bestVal = tmp;

		bestGM.action = bestAct;

		return bestGM;



	def getAction(self,b):
		act = self.Gamma[np.argmax([self.continuousDot(j,b) for j in self.Gamma])].action;
		return act;

	def getSecondaryAction(self,b,exclude):
		sG = [];
		for g in self.Gamma:
			if(g.action not in exclude):
				sG.append(g);
		act = sG[np.argmax([self.continuousDot(j,b) for j in sG])].action;
		return act;

	def getGreedyAction(self,b,x):
		cut = b.slice2DFrom4D(retGS=True,vis=False);
		MAP = cut.findMAP2D();
		cop = [x[0],x[1]];
		rob = [MAP[0],MAP[1]];
		xdist = cop[0]-rob[0];
		ydist = cop[1]-rob[1];

		if(abs(xdist)>abs(ydist)):
			if(xdist > 0):
				act = 0;
			else:
				act = 1;
		else:
			if(ydist > 0):
				act = 2;
			else:
				act = 3;

		return act;

	def MDPValueIteration(self,gen = True):
		if(gen):
			#Intialize Value function
			self.ValueFunc = copy.deepcopy(self.r);
			for g in self.ValueFunc.Gs:
				g.weight = -1000;

			comparision = GM();
			comparision.addG(Gaussian([1,0,0,0],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],1));

			uniform = GM();
			for i in range(0,5):
				for j in range(0,5):
					for k in range(0,5):
						for l in range(0,5):
							uniform.addG(Gaussian([i,j,k,l],[[4,0,0,0],[0,4,0,0],[0,0,4,0],[0,0,0,4]],1));

			count = 0;

			#until convergence
			while(not self.ValueFunc.comp(comparision) and count < 30):
				print(count);
				comparision = copy.deepcopy(self.ValueFunc);
				count += 1;
				#print(count);
				maxVal = -10000000;
				maxGM = GM();
				for a in range(0,2):
					suma = GM();
					for g in self.ValueFunc.Gs:
						mean = (np.matrix(g.mean)-np.matrix(self.delA[a])).tolist();
						var = (np.matrix(g.var) + np.matrix(self.delAVar)).tolist();
						suma.addG(Gaussian(mean,var,g.weight));
					suma.addGM(self.r);
					tmpVal = self.continuousDot(uniform,suma);
					if(tmpVal > maxVal):
						maxVal = tmpVal;
						maxGM = copy.deepcopy(suma);

				maxGM.scalerMultiply(self.discount);
				maxGM = maxGM.kmeansCondensationN(20);
				self.ValueFunc = copy.deepcopy(maxGM);

			#self.ValueFunc.display();
			#self.ValueFunc.plot2D();
			print("MDP Value Iteration Complete");
			#f = open("../policies/MDP4DIntercept.npy","w");
			#np.save(f,self.ValueFunc);
			file = "policies/MDP4DIntercept";
			self.ValueFunc.printGMArrayToFile([self.ValueFunc],file);
		else:
			#self.ValueFunc = np.load("../policies/MDP4DIntercept.npy").tolist();
			file = "policies/MDP4DIntercept";
			tmp = GM();
			self.ValueFunc = tmp.readGMArray4D(file)[0];




	def getMDPAction(self,x):
		maxVal = -10000000;
		maxGM = GM();
		bestAct = 0;
		for a in range(0,len(self.delA)):
			suma = GM();
			for g in self.ValueFunc.Gs:
				mean = (np.matrix(g.mean)-np.matrix(self.delA[a])).tolist();
				var = (np.matrix(g.var) + np.matrix(self.delAVar)).tolist();
				suma.addG(Gaussian(mean,var,g.weight));
			suma.addGM(self.r);

			tmpVal = suma.pointEval(x);
			if(tmpVal > maxVal):
				maxVal = tmpVal;
				maxGM = suma;
				bestAct = a;
		return bestAct;

	def solveQ(self):

		self.Q =[0]*len(self.delA);
		V = self.ValueFunc;
		for a in range(0,len(self.delA)):
			self.Q[a] = GM();
			for i in range(0,V.size):
				mean = (np.matrix(V.Gs[i].mean)-np.matrix(self.delA[a])).tolist();
				var = (np.matrix(V.Gs[i].var) + np.matrix(self.delAVar)).tolist()
				self.Q[a].addG(Gaussian(mean,var,V.Gs[i].weight));
			self.Q[a].addGM(self.r);
		#f = open("../policies/qmdp4DIntercept.npy","w");
		#np.save(f,self.Q);

	def getQMDPAction(self,b):
		act = np.argmax([self.continuousDot(self.Q[j],b) for j in range(0,len(self.Q))]);
		return act;

	def getQMDPSecondaryAction(self,b,exclude=[]):
		sG = [];
		for a in range(0,len(self.delA)):
			if(a not in exclude):
				sG.append(a);
		bestVal = -10000000000;
		act = -1;
		for a in sG:
			tmpVal = self.continuousDot(self.Q[a],b);
			if(tmpVal > bestVal):
				bestVal = tmpVal;
				act = a;
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
			if(beta.comp(b)):
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

	#TODO: You changed the variance for the cop
	#TODO: You changed the length of the transitions

	#movement variance is 0.25 for the robber, stationary is 0.0001
	def buildTransition(self):
		self.delAVar = [[0.0001,0,0,0],[0,0.0001,0,0],[0,0,0.15,0],[0,0,0,0.15]];
		self.delA = [[-1,0,0,0],[1,0,0,0],[0,-1,0,0],[0,1,0,0],[0,0,0,0]];

	def buildAltObs(self,gen=True):
		#A front back left right center model
		#0:center
		#1-4: left,right,down,up

		if(gen):
			self.pz = [0]*5;
			for i in range(0,5):
				self.pz[i] = GM();
			var = [[.7,0,0,0],[0,.7,0,0],[0,0,.7,0],[0,0,0,.7]];
			for i in range(-1,7):
				for j in range(-1,7):
					self.pz[0].addG(Gaussian([i,j,i,j],var,1));

			for i in range(-1,7):
				for j in range(-1,7):
					for k in range(-1,7):
						for l in range(-1,7):
							if(i-k>0):
								self.pz[1].addG(Gaussian([i,j,k,l],var,1));
							if(i-k<0):
								self.pz[2].addG(Gaussian([i,j,k,l],var,1));
							if(j-l>0):
								self.pz[3].addG(Gaussian([i,j,k,l],var,1));
							if(j-l<0):
								self.pz[4].addG(Gaussian([i,j,k,l],var,1));

			print('Plotting Observation Models');
			for i in range(0,len(self.pz)):
				self.plotAllSlices(self.pz[i],title = 'Uncondensed Observation');

			print('Condensing Observation Models');
			for i in range(0,len(self.pz)):
				self.pz[i] = self.pz[i].kmeansCondensationN(50,lowInit = [-1,-1,-1,-1], highInit = [7,7,7,7]);

			print('Plotting Condensed Observation Models');
			for i in range(0,len(self.pz)):
				self.plotAllSlices(self.pz[i],title = 'Condensed Observation');



			#f = open("../models/obsModel4DIntercept.npy","w");
			#np.save(f,self.pz);
			file = 'models/obsAltModel4DIntercept';
			self.pz[0].printGMArrayToFile(self.pz,file);
		else:
			file = 'models/obsModel4DIntercept';
			tmp = GM();
			self.pz = tmp.readGMArray4D(file);


	def buildObs(self,gen=True):
		if(gen):
			self.pz = [GM(),GM()];
			var = [[1,0,.7,0],[0,1,0,.7],[.7,0,1,0],[0,.7,0,1]];
			for i in range(-2,8):
				for j in range(-2,8):
					self.pz[0].addG(Gaussian([i,j,i,j],var,1));

			for i in range(-2,8):
				for j in range(-2,8):
					for k in range(-2,8):
						for l in range(-2,8):
							if(abs(i-k) >=2 or abs(j-l) >= 2):
								self.pz[1].addG(Gaussian([i,j,k,l],var,1));




			print('Plotting Observation Models');
			self.plotAllSlices(self.pz[0],title = 'Uncondensed Detection');
			self.plotAllSlices(self.pz[1],title = 'Uncondensed Non-Detect');


			print('Condensing Observation Models');
			self.pz[0].condense(20);

			self.pz[1] = self.pz[1].kmeansCondensationN(45,lowInit = [-1,-1,-1,-1], highInit = [7,7,7,7]);


			print('Plotting Condensed Observation Models');
			self.plotAllSlices(self.pz[0],title = 'Condensed Detection');
			self.plotAllSlices(self.pz[1],title = 'Condensed Non-Detect');



			#f = open("../models/obsModel4DIntercept.npy","w");
			#np.save(f,self.pz);
			file = '../models/obsModel4DIntercept';
			self.pz[0].printGMArrayToFile(self.pz,file);
		else:
			file = '../models/obsModel4DIntercept';
			tmp = GM();
			self.pz = tmp.readGMArray4D(file);





	def buildReward(self,gen = True):
		if(gen):
			self.r = GM();
			var = [[1,0,.7,0],[0,1,0,.7],[.7,0,1,0],[0,.7,0,1]];
			for i in range(-2,8):
				for j in range(-2,8):
					self.r.addG(Gaussian([i,j,i,j],var,5.6));

			for i in range(-2,8):
				for j in range(-2,8):
					for k in range(-2,8):
						for l in range(-2,8):
							if(abs(i-j) >=2 or abs(k-l) >= 2):
								self.r.addG(Gaussian([i,j,k,l],var,-1));

			print('Plotting Reward Model');
			self.plotAllSlices(self.r,title = 'Uncondensed Reward');

			print('Condensing Reward Model');
			self.r.condense(50);

			print('Plotting Condensed Reward Model');
			self.plotAllSlices(self.r,title = 'Condensed Reward');


			#f = open("../models/rewardModel4DIntercept.npy","w");
			#np.save(f,self.r);
			file = 'models/rewardModel4DIntercept';
			self.r.printGMArrayToFile([self.r],file);
		else:
			#self.r = np.load("../models/rewardModel4DIntercept.npy").tolist();
			file = 'models/rewardModel4DIntercept';
			tmp = GM();
			self.r = tmp.readGMArray4D(file)[0];


	def beliefUpdate(self,b,a,o,maxMix = 10):

		btmp = GM();

		for i in self.pz[o].Gs:
			for j in b.Gs:

				tmp = mvn.pdf(np.add(np.matrix(j.mean),np.matrix(self.delA[a])).tolist(),i.mean,self.covAdd(self.covAdd(i.var,j.var),self.delAVar))
				#print(i.weight,j.weight,tmp);
				w = i.weight*j.weight*tmp.tolist();

				sig = (np.add(np.matrix(i.var).I, np.matrix(self.covAdd(j.var, self.delAVar)).I)).I.tolist();

				#sstmp = np.matrix(i.var).I*np.transpose(i.mean) + np.matrix(self.covAdd(j.var + self.delAVar)).I*np.transpose(np.add(np.matrix(j.mean),np.matrix(delA[a])));
				sstmp1 = np.matrix(i.var).I*np.transpose(np.matrix(i.mean));
				sstmp2 = np.matrix(self.covAdd(j.var,self.delAVar)).I;
				sstmp21 = np.add(np.matrix(j.mean),np.matrix(self.delA[a]));


				sstmp3 = sstmp1 + sstmp2*np.transpose(sstmp21);
				smean = np.transpose(sig*sstmp3).tolist()[0];

				btmp.addG(Gaussian(smean,sig,w));


		btmp = btmp.kmeansCondensationN(maxMix);
		#btmp.condense(maxMix);
		btmp.normalizeWeights();

		return btmp;

	def distance(self,x1,y1,x2,y2):
		dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
		dist = math.sqrt(dist);
		return dist;

	def getNextPose(self,x,isCop = True,exclude = []):
		plotFlag = True;
		if(self.b == None):
			self.b = GM([x[0],x[1],2.5,2.5],[[0.01,0,0,0],[0,0.01,0,0],[0,0,5,0],[0,0,0,5]],1);
			plotFlag = False;

		prevX = copy.deepcopy(x);
		act = -1;
		if(isCop):
			z=-1;
			obsName = 'None';
			if(plotFlag):

				act = self.getQMDPSecondaryAction(self.b);
				z=-1;
				while(z not in [4,6,2,8,5,99]):
					try:
						z = int(raw_input('Observation?'));
						if(z == 99):
							break;
					except:
						if(z not in [4,6,2,8,5,99]):
							print("Please enter a valid observation...");
				if(z == 4):
					z = 1;
					obsName = 'Left';
				elif(z == 6):
					z = 2;
					obsName = 'Right';
				elif(z == 2):
					z = 3;
					obsName = 'Down';
				elif(z == 8):
					z = 4;
					obsName = 'Up';
				elif(z ==5):
					z = 0;
					obsName = 'Near';
				if(z == 99):
					z = -1;
					self.exitFlag = True;


				self.b = self.beliefUpdate(self.b,act,z);

			self.axes.cla();
			xlabel = 'X Position';
			ylabel = 'Y Position';
			title = 'Most Recent Observation: ' + obsName;

			[xx,yy,c] = self.b.slice2DFrom4D(vis = False);

			self.axes.contourf(xx,yy,c,cmap = 'viridis');

			col = 'r';
			if(z == 0):
				col = 'g'
			#cop = self.axes.scatter(x[0],x[1],color = col,s = 100);
			#robber = self.axes.scatter(x[2],x[3],color = 'b',s = 100);
			self.axes.set_xlabel(xlabel);
			self.axes.set_ylabel(ylabel);
			self.axes.set_title(title);


		if(act == -1):
			act = self.getQMDPSecondaryAction(self.b,exclude);
		#x = np.random.multivariate_normal([x[0] + self.delA[act][0],x[1] + self.delA[act][1],x[2]+self.delA[act][2],x[3]+self.delA[act][3]],self.delAVar,size =1)[0].tolist();
		x[0] = x[0] + self.delA[act][0];
		x[1] = x[1] + self.delA[act][1];
		x[2] = x[2] + self.delA[act][2] + (random.random() - 0.5);
		x[3] = x[3] + self.delA[act][3] + (random.random() - 0.5);

		x[0] = min(x[0],5);
		x[0] = max(x[0],0);
		x[1] = min(x[1],5);
		x[1] = max(x[1],0);
		x[2] = min(x[2],5);
		x[2] = max(x[2],0);
		x[3] = min(x[3],5);
		x[3] = max(x[3],0);


		if(isCop):
			col = 'r';
			if(z == 0):
				col = 'g'
			cop = self.axes.scatter(prevX[0],prevX[1],color = col,s = 100);
			robber = self.axes.scatter(prevX[2],prevX[3],color = 'b',s = 100);

			self.axes.arrow(prevX[0],prevX[1],x[0]-prevX[0],x[1]-prevX[1],head_width = 0.05,head_length=0.15, fc=col,ec=col);
			#self.axes.arrow(prevX[2],prevX[3],x[2]-prevX[2],x[3]-prevX[3],head_width = 0.05,head_length=0.25, fc='b',ec='b');
			plt.pause(0.5);

		return x;




	def simulate(self,policy = "interceptAlphasTemp.npy",initialPose = [1,1,4,4],initialBelief = None, numSteps = 20,mul = False,QMDP = False,MDP = False,mdpGen = True,human = False,greedy = False,randSim = False,altObs = False,belSave = 'tmpbelSave.npy',beliefMaxMix = 10,verbose = True):

		if(initialBelief == None):
			b = GM();
			var = [[0.01,0,0,0],[0,0.01,0,0],[0,0,4,0],[0,0,0,4]];
			b.addG(Gaussian([initialPose[0],initialPose[1],2.5,2.5],var,1));
		else:
			b = initialBelief;

		if(human):
			fig,ax = plt.subplots();
		elif(MDP or QMDP):
			self.MDPValueIteration(mdpGen);
			if(QMDP):
				self.solveQ();

		x = initialPose;
		allX = [];
		allX.append(x);
		allX0 = [];
		allX0.append(x[0]);
		allX1 = [];
		allX1.append(x[1])
		allX2 = [];
		allX2.append(x[2])
		allX3 = [];
		allX3.append(x[3])

		reward = 0;
		allReward = [0];
		allB = [];
		allB.append(b);

		allAct = [];

		if(randSim):
			for i in range(0,3):
				for j in range(0,3):
					for k in range(0,3):
						for l in range(0,3):
							x = [i*2+0.5,j*2+0.5,k*2+0.5,l*2+0.5];
							b = GM();
							var = [[0.01,0,0,0],[0,0.01,0,0],[0,0,4,0],[0,0,0,4]];
							b.addG(Gaussian([x[0],x[1],2.5,2.5],var,1));
							for h in range(0,numSteps):
								act = random.randint(0,4);
								x = np.random.multivariate_normal([x[0] + self.delA[act][0],x[1] + self.delA[act][1],x[2]+self.delA[act][2],x[3]+self.delA[act][3]],self.delAVar,size =1)[0].tolist();

								x[0] = min(x[0],5);
								x[0] = max(x[0],0);
								x[1] = min(x[1],5);
								x[1] = max(x[1],0);
								x[2] = min(x[2],5);
								x[2] = max(x[2],0);
								x[3] = min(x[3],5);
								x[3] = max(x[3],0);

								if(not altObs):
									if(self.distance(x[0],x[1],x[2],x[3]) <= 1):
										z = 0;
									else:
										z = 1;
								else:
									if(self.distance(x[0],x[1],x[2],x[3]) <= 1):
										z = 0;
									elif(x[0]-x[2] > 0 and abs(x[0]-x[2]) > abs(x[1]-x[3])):
										z = 1;
									elif(x[0]-x[2] < 0 and abs(x[0]-x[2]) > abs(x[1]-x[3])):
										z = 2;
									elif(x[1]-x[3] > 0 and abs(x[1]-x[3]) > abs(x[0]-x[2])):
										z = 3;
									elif(x[1]-x[3] < 0 and abs(x[1]-x[3]) > abs(x[0]-x[2])):
										z = 4;

								b = self.beliefUpdate(b,act,z,beliefMaxMix);





								allB.append(b);
								allX.append(x);
								allX0.append(x[0]);
								allX1.append(x[1]);
								allX2.append(x[2]);
								allX3.append(x[3]);
			f = open(belSave,"w");
			np.save(f,allB);

			#allB[numSteps].plot2D();
			print(max(allX0), min(allX0), sum(allX0) / float(len(allX0)));
			print(max(allX1), min(allX1), sum(allX1) / float(len(allX1)));
			print(max(allX2), min(allX2), sum(allX2) / float(len(allX2)));
			print(max(allX3), min(allX3), sum(allX3) / float(len(allX3)));
		else:
			self.Gamma = np.load(policy);

			for count in range(0,numSteps):

				if(human):


					ax.cla();
					col = 'b';
					if(self.distance(x[0],x[1],x[2],x[3]) <= 1):
						col = 'g'
					[xx,y,c] = b.slice2DFrom4D(vis=False);
					plt.contourf(xx,y,c,cmap = 'viridis');
					plt.scatter(x[0],x[1],c=col,s = 200);
					plt.pause(0.5);

					act = -1;
					while(act not in [4,6,2,8,5,99]):
						try:
							act = int(raw_input('Action?'));
							if(act == 99):
								break;
						except:
							print("Please enter a valid action...");
					if(act == 4):
						act = 0;
					elif(act == 6):
						act = 1;
					elif(act == 2):
						act = 2;
					elif(act == 8):
						act = 3;
					elif(act ==5):
						act = 4;
					if(act == 99):
						self.exitFlag == True;
						break;


				elif(greedy):
					act = self.getGreedyAction(b,x);
				elif(MDP):
					act = self.getMDPAction(x);
					#print(act);
				elif(QMDP):
					act = self.getQMDPAction(b);
				else:
					act = self.getAction(b);

 				if((x[0] == 0 and act == 0) or (x[0] == 5 and act == 1) or (x[1] == 0 and act == 2) or (x[1] == 5 and act == 3)):
 					act = 4;


				x = np.random.multivariate_normal([x[0] + self.delA[act][0],x[1] + self.delA[act][1],x[2]+self.delA[act][2],x[3]+self.delA[act][3]],self.delAVar,size =1)[0].tolist();

				allAct.append(act);
				x[0] = min(x[0],5);
				x[0] = max(x[0],0);
				x[1] = min(x[1],5);
				x[1] = max(x[1],0);
				x[2] = min(x[2],5);
				x[2] = max(x[2],0);
				x[3] = min(x[3],5);
				x[3] = max(x[3],0);

				if(not altObs):
					if(self.distance(x[0],x[1],x[2],x[3]) <= 1):
						z = 0;
					else:
						z = 1;
				else:
					if(self.distance(x[0],x[1],x[2],x[3]) <= 1):
						z = 0;
					elif(x[0]-x[2] > 0 and abs(x[0]-x[2]) > abs(x[1]-x[3])):
						z = 1;
					elif(x[0]-x[2] < 0 and abs(x[0]-x[2]) > abs(x[1]-x[3])):
						z = 2;
					elif(x[1]-x[3] > 0 and abs(x[1]-x[3]) > abs(x[0]-x[2])):
						z = 3;
					elif(x[1]-x[3] < 0 and abs(x[1]-x[3]) > abs(x[0]-x[2])):
						z = 4;

				if(not MDP):
					b = self.beliefUpdate(b,act,z,beliefMaxMix);

				'''
				col = 'b';
				if(self.distance(x[0],x[1],x[2],x[3]) <= 1):
					col = 'g'
				[xx,y,c] = b.slice2DFrom4D(vis=False);
				plt.contourf(xx,y,c,cmap = 'viridis');
				plt.scatter(x[0],x[1],c=col,s = 200);
				plt.pause(0.5);
				print(act);
				'''


				allB.append(b);
				allX.append(x);
				allX0.append(x[0]);
				allX1.append(x[1]);
				allX2.append(x[2]);
				allX3.append(x[3]);

				if(self.distance(x[0],x[1],x[2],x[3]) <= 1):
					reward += 3;
					allReward.append(reward);
				else:
					reward -= 1;
					allReward.append(reward);



			allAct.append(-1);
			if(verbose):
				print("Simulation Complete. Accumulated Reward: " + str(reward));
			return [allB,allX0,allX1,allX2,allX3,allAct,allReward];

	def plotRewardErrorBounds(self,allSimRewards):
		#find average reward
		averageRewards = copy.deepcopy(allSimRewards[0]);

		for i in range(1,simCount):
			for j in range(0,len(allSimRewards[i])):
				averageRewards[j] += allSimRewards[i][j];

		for i in range(0,len(averageRewards)):
			averageRewards[i] = averageRewards[i]/len(allSimRewards);

		#find sigma bounds
		sampleVariances = [0 for i in range(0,len(allSimRewards[0]))];
		twoSigmaBounds = [0 for i in range(0,len(allSimRewards[0]))];
		for i in range(0,len(sampleVariances)):
			suma = 0;
			for j in range(0,len(allSimRewards)):
				suma += (allSimRewards[j][i] - averageRewards[i])**2;
			sampleVariances[i] = suma/len(allSimRewards);
			twoSigmaBounds[i] = sqrt(sampleVariances[i])*2;
		#plot figure
		time = [i for i in range(0,len(allSimRewards[0]))];
		plt.figure();
		plt.errorbar(time,averageRewards,yerr=twoSigmaBounds);
		plt.xlabel('Simulation Step');
		plt.title('Average Simulation Reward with Error Bounds for ' + str(len(allSimRewards)) + ' simulations.');
		plt.ylabel('Reward');
		plt.show();

	def ani(self,bels,allX0,allX1,allX2,allX3,numFrames = 20):
		fig, ax = plt.subplots()
		a = np.linspace(0,0,num = 100);
		xlabel = 'Robber X Position';
		ylabel = 'Robber Y Position';
		title = 'Belief Animation';

		images = [];

		for t in range(0,numFrames):
		 	if t != 0:
				ax.cla();


				[x,y,c] = bels[t].slice2DFrom4D(vis = False);

				ax.contourf(x,y,c,cmap = 'viridis');

				col = 'b';
				if(self.distance(allX0[t],allX1[t],allX2[t],allX3[t]) <= 1):
					col = 'g'
				cop = ax.scatter(allX0[t],allX1[t],color = col,s = 100);
				robber = ax.scatter(allX2[t],allX3[t],color = 'red',s = 100);
				ax.set_xlabel(xlabel);
				ax.set_ylabel(ylabel);
				ax.set_title(title);
				fig.savefig('../tmp/img' + str(t) + ".png");
				#print('../tmp/img' + str(t) + ".png")
				plt.pause(0.5)

		for k in range(0,numFrames-1):
			fname = "../tmp/img%d.png" %k
			#print(fname);
			img = mgimg.imread(fname);
			imgplot = plt.imshow(img);
			images.append([imgplot]);


		#fig = plt.figure();
		my_ani = animation.ArtistAnimation(fig,images,interval = 20);
		my_ani.save("../Results/animation.gif",fps = 2)
		#plt.show();

	def signal_handler(self,signal, frame):
		print("Stopping Policiy Generation and printing to file");
		self.exitFlag = True;

	def plotAllSlices(self,a,title):
		fig,ax = plt.subplots(2,2);
		[x1,y1,c1] = a.slice2DFrom4D(vis=False,dims=[0,2]);
		ax[0,0].contourf(x1,y1,c1,cmap = 'viridis');
		ax[0,0].set_title('Cop X with Robber X');

		[x2,y2,c2] = a.slice2DFrom4D(vis=False,dims=[0,3]);
		ax[0,1].contourf(x2,y2,c2,cmap = 'viridis');
		ax[0,1].set_title('Cop X with Robber Y');

		[x3,y3,c3] = a.slice2DFrom4D(vis=False,dims=[1,2]);
		ax[1,0].contourf(x3,y3,c3,cmap = 'viridis');
		ax[1,0].set_title('Cop Y with Robber X');

		[x4,y4,c4] = a.slice2DFrom4D(vis=False,dims=[1,3]);
		ax[1,1].contourf(x4,y4,c4,cmap = 'viridis');
		ax[1,1].set_title('Cop Y with Robber Y');

		fig.suptitle(title);
		plt.show();

	def loadPolicy(self,fileName):
		self.Gamma = np.load(fileName);


if __name__ == "__main__":




	#Files
	belSave = '../beliefs/2dInterceptBelief3.npy';
	belLoad = '../beliefs/2dInterceptBelief2.npy';
	alsave = '../policies/2dInterceptAlphas2.npy';
	alLoad = '../policies/2dInterceptAlphas2.npy';

	'''
	********
	Alphas:
	1: Junk

	Beliefs:
	1: Who knows... not alt
	2: Alt

	********
	'''


	#Flips and switches

	#Solver Params
	sol = False;
	iterations = 500;
	discount = 0.4;
	altObs = True;

	#controls obs and reward generation
	generate = False;

	#Simulation Controls
	sim = False;

	simRand = False;
	randStep = 3;
	numStep = 100;

	humanInput = False;

	mdpPolicy = False;
	mdpGen = False;

	greedySim = False;

	qmdp = False;

	mulSim = False;
	simCount = 10;

	#ususally around 10
	belMaxMix = 10;

	hObs = True;
	hardware = True;


	a = InterceptTestGenerator(beliefFile = belLoad,dis = discount,gen = generate,altObs = altObs,qGen = True,humObs = hObs);
	signal.signal(signal.SIGINT, a.signal_handler);


	if(hardware):
		x = [1,1,4,3];
		for i in range(0,20):
			#a.getHumanObservation();
			x = a.getNextPose(x,True);
			if(a.exitFlag):
				break;
			#print(x);




	if(sol):
		a.solve(N = iterations,alsave = alsave,verbose = True);
	if(sim):
		if(not simRand and not mulSim):
			inPose = [random.randint(0,5),random.randint(0,5),random.randint(0,5),random.randint(0,5)];

			[allB,allX0,allX1,allX2,allX3,allAct,allReward] = a.simulate(policy = alLoad,initialPose = inPose,numSteps = numStep,belSave = belSave,QMDP = qmdp,MDP = mdpPolicy,mdpGen =mdpGen,human = humanInput,greedy=greedySim,randSim = simRand,altObs = altObs,beliefMaxMix = belMaxMix);



			#plt.plot(allReward);

			#plt.show();
			dist = [];
			for i in range(0,len(allX0)):
				dist.append(a.distance(allX0[i],allX1[i],allX2[i],allX3[i]));
			fig,ax = plt.subplots(2,sharex = True);
			x = [i for i in range(0,len(allX0))];

			ax[0].plot(x,allX0,label='Cop X');
			ax[0].plot(x,allX2,label = 'Robber X');
			ax[0].set_ylim([0,5]);
			ax[1].plot(x,allX1,label = 'Cop Y');
			ax[1].plot(x,allX3,label = 'Robber Y');
			ax[1].set_ylim([0,5]);

			#ax.legend( loc='upper right' )

			plt.show();

			plt.plot(dist);
			axes = plt.gca();
			axes.set_ylim([-10,10]);
			plt.axhline(y=1, xmin=0, xmax=100, linewidth=2, color = 'k')
			plt.axhline(y=-1, xmin=0, xmax=100, linewidth=2, color = 'k')
			plt.xlabel('Simulation Step');
			plt.ylabel('Difference: Robber-Cop');
			plt.title('Robot Position Difference with detection zones');
			plt.show();


			a.ani(allB,allX0,allX1,allX2,allX3,numFrames = numStep);

			'''
			for i in range(0,len(allX0)):
				print(allX0[i],allX1[i]);
				print(allAct[i]);
				if(abs(allX0[i]-allX1[i]) <= 1):
					print("Detection");
			'''

		elif(not simRand):
			#run simulations
			allSimRewards = [];
			for i in range(0,simCount):
				inPose = [random.randint(0,5),random.randint(0,5),random.randint(0,5),random.randint(0,5)];
				print("Starting simulation: " + str(i+1) + " of " + str(simCount) + " with initial position: " + str(inPose));
				[allB,allX0,allX1,allX2,allX3,allAct,allReward] = a.simulate(policy = alLoad,initialPose = inPose,numSteps = numStep,greedy = greedySim,human = humanInput, mul = mulSim,QMDP = qmdp,MDP = mdpPolicy,mdpGen =mdpGen,belSave = belSave,randSim = simRand,altObs = altObs,beliefMaxMix = belMaxMix,verbose = False);
				allSimRewards.append(allReward);
				print("Simulation complete. Reward: " + str(allReward[numStep-1]));
			a.plotRewardErrorBounds(allSimRewards);


		elif(simRand):
			a.simulate(policy = alLoad,initialPose = [1,1,4,4],numSteps = randStep,belSave = belSave,randSim = simRand,altObs = altObs,beliefMaxMix = belMaxMix);
