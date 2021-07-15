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

		particles_dict = {
			'px':px,
			'py':py,
			'pz':pz,
			'E' :E,
			'eta':eta,
			'phi':phi,
			'PT':pt
			}

		return pd.DataFrame(data=particles_dict)
	'''
	def affinity(self,particle_1,particle_2):
		Function that calculates affinity between two particles.
		The affinity has a gaussian shape and it requires the distance between particles.
		Parameters such as alpha and sigma are needed.
		
		Input: two 1d arrays, i.e. two particles whose entries are their coordinates in real space.
		(Remember that the angle is a cyclic variable).

		Output: a single number, the affinity.
		phi_dist = Components.angular_distance(particle_1[1],particle_2[1])
		rapidity_dist = particle_1[0] - particle_2[0]
		exponent = np.sqrt(phi_dist**2 + rapidity_dist**2)
		return np.exp(-exponent**self.alpha/self.sigma)


	def affinity_matrix(self,particles):
		The function compute the affinity matrix.
		For each pair of particles it computes the affinity.

		Input: array (list of particles).

		Output: a symmetric 2d array (diagonal filled with zeros).
		eta,phi = particles['eta'].values,particles['phi'].values
		all_p = np.array([[eta[i],phi[i]] for i in range(len(eta))])
		pairs = [*combinations(all_p,2)]
		upp_t = [self.affinity(i[0],i[1]) for i in pairs]
		length = len(all_p)
		indeces_upp_t = np.triu_indices(length,1)
		aff_m = np.zeros((length,length))
		aff_m[indeces_upp_t] = upp_t
		aff_m_t = np.transpose(aff_m)
		return aff_m + aff_m_t
	'''

	def affinity_matrix(self,particles):
		eta,phi = particles['eta'].values,particles['phi'].values
		phi_col = phi[...,None]
		eta_col = eta[...,None]

		ETA = eta - eta_col
		PHI = phi - phi_col

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

	def laplacian(self,affinity_m,weights):
		'''
		The function computes the normalized laplacian.
		First it computes the unnormalized one as l=weight-aff_m, then it will be normalized.
		The weights are transformed into a 2d array.

		Input: one 2d array (affinity matrix) and a 1d array (weights).

		Output: normalized laplacian.
		'''
		weights = np.array(weights)
		weights_m = np.diag(weights**-0.5)
		weights_m2 = np.diag(np.sum(affinity_m,axis=1))
		l = weights_m2 - affinity_m
		return np.matmul(weights_m,np.matmul(l,weights_m))

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
		evals, evecs = la.eigh(lapl)
		evecs = np.transpose(evecs)
		
		evals, evecs = evals[1:],evecs[1:]
		np.clip(evals,0.001,None,out=evals)

		e_space=[]
		for i,val in enumerate(evals):
			if val<self.eval_min:
				e_space.append(evecs[i]/(evals[i]**self.eval_exp))

		evals_nn = int(np.sum(evals < self.eval_min))
		evals = evals**self.eval_exp
		evecs /= evals	
		print(evals_nn)	
		E,V = evals[:evals_nn],evecs[:evals_nn]
		print(len(V),'-',len(e_space))
		'''
		evals, evecs = la.eigh(lapl)
		'''	
		#check wheter the first eigenvalue is zero
		compare_zero = np.zeros_like(evecs[0])
		compare_eig0 = evals[0]*evecs[0]
		compare = np.isclose(compare_zero,compare_eig0)
		if compare.any()==False:
			print('Ops')
		else:
			print('Yeah')
		'''
		np.clip(evals,0.001,None,out=evals)
		eval_nn = int(np.sum(evals<self.eval_min))
		evals,evecs = evals[1:eval_nn], evecs[:,1:eval_nn]
		evals = evals**self.eval_exp
		evecs/=evals
		#evecs = np.transpose(evecs)
		return evals,evecs

	'''
	def particles_in_es(self,emb_space):
		return np.transpose(emb_space)

	def combine(self,particles_list):
		comb = [*combinations(particles_list,2)]
		for i in range(len(comb)):
			comb[i] = list(comb[i])
		return comb
	'''
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


	def old_to_new(self,old_1,old_2):
		new_momentum = [old_1[ind]+old_2[ind] for ind in range(4)]
		new_phi,new_pt = Components.pxpy_to_phipt(new_momentum[0],new_momentum[1])
		new_eta = Components.ptpze_to_rapidity(new_pt,new_momentum[2],new_momentum[3])
		print(new_pt)
		
		new_momentum.extend((new_eta,new_phi,new_pt))
		return np.array(new_momentum)


	def merge(self,particles,angular_d):
		np.fill_diagonal(angular_d,np.inf)
		row,column = np.unravel_index(np.argmin(angular_d),np.shape(angular_d))
		
		old_1,old_2 = particles.iloc[row],particles.iloc[column]
		new_p = self.old_to_new(old_1,old_2)
		
		particles.drop([particles.index[row],particles.index[column]],inplace=True)
		particles.loc[min(particles.index)-1] = new_p
		return particles,row,column

	def update_weights(self,weights_v,index_a,index_b):
		w_1,w_2 = weights_v[index_a], weights_v[index_b]
		
		weights_v = np.delete(weights_v,[index_a,index_b])	
		weights_v = np.append(weights_v,[w_1+w_2])
		return  weights_v

	def update_affinity_m(self,particles,aff_m,index_a,index_b):
		aff_m = np.delete(aff_m,index_a,axis=0)
		aff_m = np.delete(aff_m,index_a,axis=1)
		aff_m = np.delete(aff_m,index_b-1,axis=0)
		aff_m = np.delete(aff_m,index_b-1,axis=1)

		eta,phi = particles['eta'].values, particles['phi'].values

		phi_dist = Components.angular_distance(phi[-1],phi[:-1])
		eta_dist = Components.angular_distance(eta[-1],eta[:-1])
		exponent = np.sqrt(phi_dist**2 + eta_dist**2)
		new_column = np.exp(-exponent**self.alpha/self.sigma)
 
		#new_column = np.array([self.affinity([eta[-1],phi[-1]],[eta[r],phi[r]]) for r in range(len(eta)-1)])
		aff_m = np.column_stack((aff_m,new_column))
		new_row = np.append(new_column,[0])
		aff_m = np.vstack((aff_m,new_row))
		return aff_m

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


	def process(self,particles,aff_m=None,weights=None,index_a=None,index_b=None,angular=None,first_step=True):
		if first_step == True:
			aff_m = self.affinity_matrix(particles)
			sel = self.selecting_neighbours(aff_m)
			weights = self.weights_vector(sel)
			laplacian = self.laplacian(sel,weights)
			val,emb_space = self.embedding_space(laplacian)
			angular = self.angular_distance(emb_space)
			index_a,index_b = 1,1	
			#particles,index_a,index_b = self.merge(particles,angular)	

	
		else:
	
			particles,index_a,index_b = self.merge(particles,angular)
			aff_m = self.update_affinity_m(particles,aff_m,index_a,index_b)
			sel = self.selecting_neighbours(aff_m)
			weights = self.update_weights(weights,index_a,index_b)
			laplacian = self.laplacian(sel,weights)
			val,emb_space = self.embedding_space(laplacian)
			angular = self.angular_distance(emb_space)
			#particles,index_a,index_b = self.merge(particles,angular)

		mean_distance = self.stop_condition(angular)
		floats = np.array([index_a,index_b,mean_distance,len(emb_space)])
		return particles,floats,aff_m,weights,angular


								
	def JETS(self,ew):
		particles = self.particles_in_rs(ew)
		tree = np.array([range(len(particles['px'].values))])

		particles,floats,aff_m,weights,angular= self.process(particles)

		c=0
		index_a,index_b = int(floats[0]),int(floats[1])
		indices = particles.index.values
		print(floats[2])
		while floats[2] < self.R and floats[3]>1:
			#print(c)
			c+=1
			#index_a,index_b = int(floats[0]),int(floats[1])
			#print(floats[2],' doing it ',index_a,index_b)
			particles,floats,aff_m,weights,angular = self.process(particles,aff_m,weights,index_a,index_b,angular,first_step=False)
			index_a,index_b = int(floats[0]),int(floats[1])
			#indices = particles.index.values

			#tree = self.history(tree,indices[index_a],indices[index_b])

			#print(floats[2],' doing it ',index_a,index_b)
			
		return particles,tree


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
	cProfile.run('main()','output.dat')

	import pstats
	from pstats import SortKey

	with open('output_time.txt','w') as f:
		p = pstats.Stats('output.dat',stream=f)
		p.sort_stats('time').print_stats()

	with open('output_calls.txt','w') as f:
		p = pstats.Stats('output.dat',stream=f)
		p.sort_stats('calls').print_stats()
