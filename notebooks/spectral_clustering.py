import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as ssd
from itertools import combinations
from statistics import mean
from jet_tools import Components


class SpectralClustering():
    def __init__(self,alpha=2.,sigma=0.15,k_nn=5,eval_min=0.4,eval_exp=1.4,R=1.26):
        self.alpha = alpha
        self.sigma = sigma
        self.k_nn = k_nn
        self.eval_min = eval_min
        self.R = R
        self.eval_exp = eval_exp
    
    def particles_in_rs(self,phi,rapidity,pt,px,py,pz,pe):
        p=[phi,rapidity,pt,px,py,pz,pe]
        return np.transpose(p)

    def affinity(self,particle_1,particle_2):
        '''
        Function that calculates affinity between two particles.
        The affinity has a gaussian shape and it requires the distance between particles.
        Parameters such as alpha and sigma are needed.
        
        Input: two 1d arrays, i.e. two particles whose entries are their coordinates in real space.
        (Remember that the angle is a cyclic variable).

        Output: a single number, the affinity.
        '''
        phi_dist = Components.angular_distance(particle_1[0],particle_2[0])
        rapidity_dist = particle_1[1] - particle_2[1]
        exponent = np.sqrt(phi_dist**2 + rapidity_dist**2)
        return np.exp(-exponent**self.alpha/self.sigma)



    def affinity_matrix(self,all_p):
        '''
        The function compute the affinity matrix.
        For each pair of particles it computes the affinity.

        Input: array (list of particles).

        Output: a symmetric 2d array (diagonal filled with zeros).
        '''
        pairs = [*combinations(all_p,2)]
        upp_t = [self.affinity(i[0],i[1]) for i in pairs]
        length = len(all_p)
        indeces_upp_t = np.triu_indices(length,1)
        aff_m = np.zeros((length,length))
        aff_m[indeces_upp_t] = upp_t
        aff_m_t = np.transpose(aff_m)
        return aff_m + aff_m_t


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
        '''
        weights=[] 
        for rows in selected_neighbours:
            weights.append(sum(rows))
        
        return  np.array(weights)

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
        weights_m2 = np.diag(weights)
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
        '''
        evals, evecs = la.eigh(lapl)
        evecs = np.transpose(evecs)
        
        #print(evals[0:2],evals[-1])
        evals, evecs = evals[1:],evecs[1:]
        
        np.clip(evals,0.001,None,out=evals)

        e_space=[]
        for i,val in enumerate(evals):
            if val<self.eval_min:
                e_space.append(evecs[i]/(evals[i]**self.eval_exp))

        #print(len(e_space))
        return evals,e_space


    def particles_in_es(self,emb_space):
        return np.transpose(emb_space)

    def combine(self,particles_list):
        comb = [*combinations(particles_list,2)]
        for i in range(len(comb)):
            comb[i] = list(comb[i])
        return comb

    def angular_distance(self,particles_list):
        '''
        The function calculates the angular distance between particles in the embedding space.

        Input: array of all the particles in the embedding space.

        Output: 1d array of all the angular distances.
        '''
        #print(np.shape(particles_list))
        return  np.arccos(np.abs(1 - ssd.pdist(particles_list,'cosine')))
        
    
    def old_to_new(self,p1_old,p2_old):
        px_new = p1_old[3] + p2_old[3]
        py_new = p1_old[4] + p2_old[4]
        pz_new = p1_old[5] + p2_old[5]
        e_new = p1_old[6] + p2_old[6]
        phi_new,pt_new = Components.pxpy_to_phipt(px_new,py_new)
        rapidity_new = Components.ptpze_to_rapidity(pt_new,pz_new,e_new)

        return np.array([phi_new,rapidity_new,pt_new,px_new,py_new,pz_new,e_new])

 
    def merge(self,p_in_rs,p_in_es,all_pairs,angular_d):
        pair = all_pairs[np.argmin(angular_d)]
        p_1,p_2 = pair[0],pair[1]
        i,j = np.where(p_in_es==p_1),np.where(p_in_es==p_2)
        i,j = i[0][0],j[0][0] 
        p1_old,p2_old = p_in_rs[i],p_in_rs[j]

        p_new = self.old_to_new(p1_old,p2_old)

        p_in_rs = list(p_in_rs)
        p_in_rs.pop(i)
        p_in_rs.pop(j-1)
        p_in_rs.append(p_new)
        p_in_rs = np.array(p_in_rs)

        return p_in_rs,i,j

    def update_weights(self,weights_v,i,j):
         w_1,w_2 = weights_v[i], weights_v[j]
         weights_v = list(weights_v)
         weights_v.remove(w_1),weights_v.remove(w_2)
         weights_v.append(w_1+w_2)
         return  np.array(weights_v)



    
    def JETS(self,phi,rapidity,pt,px,py,pz,e):
        particles_in_rs = self.particles_in_rs(phi,rapidity,pt,px,py,pz,e)
        aff_m = self.affinity_matrix(particles_in_rs)
        sel = self.selecting_neighbours(aff_m)
        weights_v = self.weights_vector(sel)
        laplacian = self.laplacian(sel,weights_v)
        val,emb_space = self.embedding_space(laplacian)
        particles_in_es = self.particles_in_es(emb_space)
        all_pairs = self.combine(particles_in_es)
        angular_d = self.angular_distance(particles_in_es)
        mean_angular_d = mean(angular_d)

        print(mean_angular_d, mean_angular_d < self.R)
        while mean_angular_d < self.R and len(emb_space)>1: 
            particles_in_rs,i,j = self.merge(particles_in_rs,particles_in_es,all_pairs,angular_d)
            weights_v = self.update_weights(weights_v,i,j) 
            aff_m = self.affinity_matrix(particles_in_rs)
            sel = self.selecting_neighbours(aff_m)
            laplacian = self.laplacian(sel,weights_v)
            val,emb_space = self.embedding_space(laplacian)
            particles_in_es = self.particles_in_es(emb_space)
            all_pairs = self.combine(particles_in_es)
            angular_d = self.angular_distance(particles_in_es)
            mean_angular_d = mean(angular_d)
             
            print(mean_angular_d,' doing it',i,j)
            print('Number of particles remained: ',len(particles_in_rs))

        print('done')
        return particles_in_rs












