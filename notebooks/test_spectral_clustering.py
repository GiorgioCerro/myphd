from spectral_clustering import SpectralClustering
from numpy import testing
import numpy as np

'''
def test_particles_in_rs():
    test = SpectralClustering()
    phi = [1,2,3,4,5]
    rapidity = [1,2,3,4,5]
    pt = [1,2,3,4,5]
    output = test.particles_in_rs(phi,rapidity,pt)
    expected_output = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
    testing.assert_allclose(output,expected_output)
'''

def test_affinity():
    test = SpectralClustering()
    p1 = np.array([1,1,2,5])
    p2 = np.array([1,1,3,0])
    output = test.affinity(p1,p2)
    expected_output = 1
    testing.assert_allclose(output,expected_output)
    
    p1 = np.array([np.pi,2.32])
    p2 = np.array([-np.pi,2.32])
    output = test.affinity(p1,p2)
    expected_output = 1 
    testing.assert_allclose(output,expected_output)


def test_aff_m():
    test=SpectralClustering()
    particles = [[1,1,1],[1,1,1]]
    m = test.affinity_matrix(particles)
    testing.assert_allclose(m,[[0,1],[1,0]])

def test_selecting_neighbours():
    test=SpectralClustering(k_nn=2)
    aff_m=np.array([[1,2,3],[2,2,1],[3,1,2]])
    out_put = test.selecting_neighbours(aff_m)
    exp_output = [[0,2,3],[2,2,0],[3,0,2]]
    testing.assert_allclose(out_put,exp_output)

def test_weights_vector():
    test=SpectralClustering()
    aff_m = [[1]]
    w = test.weights_vector(aff_m)
    testing.assert_allclose(w,[1])
    
    aff_m = [[1,2,3],[2,3,1],[0.1,5,2]]
    w=test.weights_vector(aff_m)
    testing.assert_allclose(w,[6,6,7.1])

def test_laplacian():
    test=SpectralClustering()
    aff_m = np.array([[1.]])
    w = np.array([1.])
    L = test.laplacian(aff_m,w) 
    testing.assert_allclose(L,[[0.]])

    aff_m = np.array([[0,1,2],[1,0,2],[2,2,0]])
    w = [3,3,4]
    L = test.laplacian(aff_m,w)
    p = 1./3.
    q = 2/np.sqrt(12)
    testing.assert_allclose(L,[[1,-p,-q],[-p,1,-q],[-q,-q,1]])


def test_update_weights():
    test=SpectralClustering()
    w = [1,2,3,4,5,6,7,8]
    w = test.update_weights(w,1,5)
    expected_output = [1,3,4,5,7,8,8]
    testing.assert_allclose(w,expected_output)

    w = test.update_weights(w,0,3)
    expected_output = [3,4,7,8,8,6]
    testing.assert_allclose(w,expected_output)
