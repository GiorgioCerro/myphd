import numpy as np
from lhereader import LHEReader
from jet_tools import Components


def distance(phi_1,rap_1,phi_2,rap_2):
    phi_d = Components.angular_distance(phi_1,phi_2)
    rap_d = rap_1 - rap_2
    return np.sqrt(phi_d**2 + rap_d**2)

def read_lhe(PATH,parent):
    reader = LHEReader(PATH)

    events =[]
    for e in reader:
        events.append(e.particles)

    px1,py1,pz1,pe1 = [],[],[],[]
    px2,py2,pz2,pe2 = [],[],[],[]
    for ev in events:
         if parent == 'g':
            n,m = 3,4
         elif parent == 'h':
            n,m = 4,5

         px1.append(ev[n].px)
         py1.append(ev[n].py)
         pz1.append(ev[n].pz)
         pe1.append(ev[n].energy)

         px2.append(ev[m].px)
         py2.append(ev[m].py)
         pz2.append(ev[m].pz)
         pe2.append(ev[m].energy)


    px1,py1,pz1,pe1 = np.array(px1),np.array(py1),np.array(pz1),np.array(pe1)
    px2,py2,pz2,pe2 = np.array(px2),np.array(py2),np.array(pz2),np.array(pe2)

    phi1,pt1 = Components.pxpy_to_phipt(px1,py1)
    rapidity1 = Components.ptpze_to_rapidity(pt1,pz1,pe1)

    phi2,pt2 = Components.pxpy_to_phipt(px2,py2)
    rapidity2 = Components.ptpze_to_rapidity(pt2,pz2,pe2)


    return phi1,rapidity1,phi2,rapidity2


def find_close_events(PATH1,PATH2):
    phi_1g,rapidity_1g,phi_2g,rapidity_2g = read_lhe(PATH1,'g')
    phi_1h,rapidity_1h,phi_2h,rapidity_2h = read_lhe(PATH2,'h')
    dist_lim = 0.02
    event_list =[]

    for i in range(10000):
        if i%1000==0:
            print(i)
        for j in range(10000):
            dist_b = distance(phi_1g[i],rapidity_1g[i],phi_1h[j],rapidity_1h[j])
            dist_ab = distance(phi_2g[i],rapidity_2g[i],phi_2h[j],rapidity_2h[j])
            
            if dist_b < dist_lim and dist_ab < dist_lim:
                event_list.append([i,j])

    return np.array(event_list)


def return_events():
    PATH1 = '/scratch/gc2c20/data/restricted_kinematics_gz/Events/run_01/unweighted_events.lhe'
    PATH2 = '/scratch/gc2c20/data/restricted_kinematics_hz/Events/run_01/unweighted_events.lhe'

    evs = find_close_events(PATH1,PATH2)

    return evs
