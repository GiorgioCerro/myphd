from jet_tools import Components
from spectral_clustering import SpectralClustering
from statistics import mean

def hope():
    PATH = '/home/gc2c20/myproject/jetTools/mini_data/mini.hepmc.awkd'

    ew = Components.EventWise.from_file(PATH)
    ew.selected_index = 0 
    phi,rapidity,pt = ew.JetInputs_Phi, ew.JetInputs_Rapidity, ew.JetInputs_PT
    px,py,pz,pe = ew.JetInputs_Px, ew.JetInputs_Py, ew.JetInputs_Pz, ew.JetInputs_Energy

    test = SpectralClustering()

    particles = test.particles_in_rs(phi,rapidity,pt,px,py,pz,pe)
    aff_m = test.affinity_matrix(particles)
    sel = test.selecting_neighbours(aff_m)
    weights = test.weights_vector(sel)

    L = test.laplacian(sel,weights)

    v = test.embedding_space(L)

    particles_in_es = test.particles_in_es(v)
    ang = test.angular_distance(particles_in_es)

    print(mean(ang))
    return(L,v,ang)

