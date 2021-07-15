import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as ssd
from itertools import combinations
#from statistics import mean
from jet_tools import Components,FormJets
import matplotlib.pyplot as plt
from join_shower import JoinShower
import pandas as pd
#from numba import jit
###UPDATE MASK MANUALLY AT THE MOMENT
class SpectralClustering():
	def __init__(self,alpha=2.,sigma=0.15,k_nn=5,eval_min=0.4,eval_exp=1.4,R=1.26):
		self.alpha = alpha
		self.sigma = sigma
		self.k_nn = k_nn
		self.eval_min = eval_min
		self.R = R
		self.eval_exp = eval_exp

	def particles_in_rs(self,ew):
		#em_shower = JoinShower.finding_children(ew,23)
		#finals = ew.JetInputs_SourceIdx
		#leaf = list(set(finals) - set(em_shower))
		#phi,rapidity,pt = ew.Phi[leaf],ew.Rapidity[leaf],ew.PT[leaf]
		#px,py,pz,e = ew.Px[leaf],ew.Py[leaf],ew.Pz[leaf],ew.Energy[leaf]

		eta,phi,pt = ew.JetInputs_Rapidity,ew.JetInputs_Phi,ew.JetInputs_PT
		px,py,pz,E = ew.JetInputs_Px,ew.JetInputs_Py,ew.JetInputs_Pz,ew.JetInputs_Energy
		indices = np.array(range(len(eta)))
		return np.transpose([px,py,pz,E,eta,phi,pt]), np.ones(len(eta),dtype=bool),indices


	def affinity_matrix(self,particles):
		eta,phi = particles[:,4],particles[:,5]
		phi_col = phi[...,None]
		eta_col = eta[...,None]

		#ooooooops check phi dist
		ETA = eta - eta_col
		PHI = Components.angular_distance(phi,phi_col)

		exponent = np.sqrt(ETA**2 + PHI**2)
		matrix =  np.exp(-exponent**self.alpha/self.sigma) 
		np.fill_diagonal(matrix,0)
		return matrix

	def selecting_neighbours(self,affinity_m):
		'''
		The function select the nearest neighbours for each particle.
		The parameter k_nn sets how many neighbours each particle should have at least.
		The graph is a fully connected one. 

		Input: symmetric 2d array (affinity matrix).

		Ouput: symmetric 2d array.
		'''
		matrix = affinity_m.copy()
		neighbour_mask = np.zeros_like(matrix,dtype = bool) 

		for row,item in enumerate(matrix):
			indeces = np.argsort(item)[-self.k_nn:]
			neighbour_mask[row,indeces]=True 

		matrix[~neighbour_mask] = 0 
		sym = np.maximum(matrix,matrix.transpose())
		return sym

	def weights_vector(self,selected_neighbours):
		'''
		The function calculates the weight for each particle.
		The weight is defined as the sum of all the affinities of the connected particles.

		Input: symmetric 2d array (affinity matrix -possibly with the selected neighbours-).

		Output: a 1d array.
		weights=[] 
		for rows in selected_neighbours:
			weights.append(sum(rows))
		'''
		return	np.sum(selected_neighbours,axis=1)

	def laplacian(self,affinity_m,weights,mask=None):
		'''
		The function computes the normalized laplacian.
		First it computes the unnormalized one as l=weight-aff_m, then it will be normalized.
		The weights are transformed into a 2d array.

		Input: one 2d array (affinity matrix) and a 1d array (weights).

		Output: normalized laplacian.
		'''
		if mask is not None:
			weights_m = np.diag(weights[mask]**-0.5)
			weights_m2 = np.diag(np.sum(affinity_m[mask][:,mask],axis=1))
			l = weights_m2 - affinity_m[mask][:,mask]
			return np.matmul(weights_m,np.matmul(l,weights_m))
	
		else:
			weights_m = np.diag(weights**-0.5)
			weights_m2 = np.diag(np.sum(affinity_m,axis=1))
			l = weights_m2 - affinity_m
			return  np.matmul(weights_m,np.matmul(l,weights_m))

	def embedding_space(self,lapl):
		'''
		The function creates the embedding space.
		First, it computes the eigenvalues and eigenvectors of the laplacian.
		It discards the first value which is referred to the trivial solution.
		It keeps only the evectors whose evalues are less than the parameter eval_min.
		The evals are clipped (values lower than 0.001 are promoted to 0.001).
		Evectors are stretched by their evalues to the power of -eval_exp.

		Input: a symmetric 2d array (laplacian).

		Output: evals and an array (the embedding space, i.e. the stretched survived evecs).
		'''
		evals, evecs = la.eigh(lapl)
		vec_zero = np.transpose(evecs)
		#eigenvalues = np.copy(evals)
		np.clip(evals,0.001,None,out=evals)
		eval_nn = int(np.sum(evals<self.eval_min))
		evals,evecs = evals[1:eval_nn], evecs[:,1:eval_nn]
		evals = evals**self.eval_exp
		evecs/=evals
		#evecs = np.transpose(evecs)
		return evals,evecs,vec_zero[0]

	def angular_distance(self,particles_list):
		'''
		The function calculates the angular distance between particles in the embedding space.

		Input: array of all the particles in the embedding space.

		Output: 1d array of all the angular distances.
		'''
		#particles_list = np.transpose(embedding_space)
		return	np.arccos(1 - ssd.cdist(particles_list,particles_list,'cosine'))
		

	def stop_condition(self,angular_dist):
		dist = angular_dist[np.tril_indices_from(angular_dist,-1)]
		return np.mean(np.sqrt(dist))

	def merge(self,particles,angular_d,mask):
		np.fill_diagonal(angular_d,np.inf)
		row,col = np.unravel_index(np.argmin(angular_d),np.shape(angular_d))
		
		#stat1
		angular_distance = angular_d[row][col]

		row = np.where(mask==True)[0][row]
		col = np.where(mask==True)[0][col]

		#stat2
		dist_phi = particles[row][4] - particles[col][4]
		dist_eta = Components.angular_distance(particles[row][5],particles[col][5]) 
		delta_R = np.sqrt(dist_phi**2+dist_eta**2 )
	
		#stat3
		pt_max = max(particles[row][-1],particles[col][-1])
		pt_min = min(particles[row][-1],particles[col][-1])
		pt_ratio = pt_min/pt_max

		particles[row] = particles[row]+particles[col]
		new_phi,new_pt = Components.pxpy_to_phipt(particles[row][0],particles[row][1])
		new_eta = Components.ptpze_to_rapidity(new_pt,particles[row][2],particles[row][3])
		'''
		min_index = np.min(particles[:,-1])
		print(min_index,'-',min_index-1)
		new_ind = min_index - 1.
		'''
		particles[row][4:] = [new_eta,new_phi,new_pt]	
		particles[col] = 0.	

		return particles,row,col,[angular_distance,delta_R,pt_ratio]

	'''
	NEED TO UNDERSTAND HOW TO UPDATE WEIGHTS AND AFFINITY!!!!!!!
	'''
	def update_weights(self,weights,index_a,index_b):
		weights_v = np.copy(weights)
		weights_v[index_a] = weights_v[index_a]+ weights_v[index_b]
		weights_v[index_b] = 0.

		return  weights_v

	def update_affinity_m(self,particles,aff,index_a,index_b):
		aff_m = np.copy(aff)
		eta,phi = particles[:,4], particles[:,5]

		phi_dist = Components.angular_distance(phi[index_a],phi)
		eta_dist = eta[index_a] - eta
		exponent = np.sqrt(phi_dist**2 + eta_dist**2)
		aff_m[index_a] = aff_m[:,index_a] = np.exp(-exponent**self.alpha/self.sigma)
		aff_m[index_a][index_a] = 0
		
		aff_m[index_b] = aff_m[:,index_b] = 0.
 
		return aff_m

	'''
	def history(self,tree,i,j):
		copy = np.copy(tree[-1])
		minimo = min(copy)
		i,j = int(i),int(j)
		ind_a = np.where(copy==copy[i])
		ind_b = np.where(copy==copy[j])
		copy[ind_a],copy[ind_b] = minimo-1,minimo-1
		copy = [copy]
		tree = np.append(tree,copy,axis=0)
		return tree

	def update_tree(self,indices,row):
		indices[row] = min(indices)-1
		return indices
			
	'''								
	def JETS(self,ew):
		#evalues,dist=[],[]
		stat = []
		particles,mask,indices = self.particles_in_rs(ew)
		aff_m = self.affinity_matrix(particles)
		sel = self.selecting_neighbours(aff_m)
		weight = self.weights_vector(sel)
		laplacian = self.laplacian(sel,weight)
		evals,evecs,vec_zero = self.embedding_space(laplacian)
		angular = self.angular_distance(evecs)
		
		mean_distance = self.stop_condition(angular)
		c=1
		col=0
		#evalues.append(evals)
		#dist.append(mean_distance)
		stat.append(vec_zero)
		while mean_distance<self.R:
			#print(c)
			particles,row,col,stat_list = self.merge(particles,angular,mask)
			weight = self.update_weights(weight,row,col)
			aff_m = self.update_affinity_m(particles,aff_m,row,col)
			sel = self.selecting_neighbours(aff_m)
			mask[col] = False
			laplacian = self.laplacian(sel,weight,mask)
			evals,evecs,vec_zero = self.embedding_space(laplacian)
			angular = self.angular_distance(evecs)
			#print(self.stop_condition(angular),'-',row,'-',col)
			mean_distance = self.stop_condition(angular)	
			print(mean_distance,'-',row,'-',col)
			
			c+=1
		
			#evalues.append(evals)
			#dist.append(mean_distance)
			stat.append(vec_zero)
		mask[col]=False
		#return particles[mask,evalues,dist
		return particles[mask],stat

test = SpectralClustering()
PATH = '../data/gz.hepmc0.hepmc.parquet'
ew = Components.EventWise.from_file(PATH)
ew.selected_index = 0


spectral_jet_params = dict(ExpofPTMultiplier=0,
	ExpofPTPosition='input',
	ExpofPTFormat='Luclus',
	NumEigenvectors=np.inf,
	StoppingCondition='meandistance',
	EigNormFactor=1.4,
	EigenvaluesLimit=0.4,
	Laplacien='symmetric',
	DeltaR=1.26,
	AffinityType='exponent',
	AffinityExp=2.,
	Sigma=0.15,
	CombineSize='sum',
	EigDistance='abscos',
	CutoffKNN=5,
	CutoffDistance=None,
	PhyDistance='angular')

def main():
	#FormJets.SpectralFull(ew,assign=True,**spectral_jet_params)
	test.JETS(ew)
if __name__ == '__main__':
	import cProfile
	cProfile.run('main()','output2.dat')

	import pstats
	from pstats import SortKey

	with open('output2_time.txt','w') as f:
		p = pstats.Stats('output2.dat',stream=f)
		p.sort_stats('time').print_stats()

	with open('output2_calls.txt','w') as f:
		p = pstats.Stats('output2.dat',stream=f)
		p.sort_stats('calls').print_stats()
