from jet_tools import ReadHepmc,PDGNames,Components,FormJets,TrueTag
import numpy as np
import awkward
from join_shower import JoinShower
import matplotlib.pyplot as plt
from statistics import mean
from scipy import stats
from sklearn import mixture

def create_awkd(PATH):
		ew = ReadHepmc.Hepmc(PATH)
		Components.add_all(ew)

		#importing charge
		identities = PDGNames.Identities()
		charges = []
		n_events = len(ew.MCPID)
		charge_dict = identities.charges
		for event_n in range(n_events):
				if event_n%100 == 0:
						print(f"{event_n/n_events:.0}", end='\r')
				ew.selected_index = event_n
				charges.append([])
				for pid in ew.MCPID:
						try:
								charges[-1].append(charge_dict[pid])
						except KeyError:
								charges[-1].append(np.nan)

		charges = awkward.fromiter(charges)
		ew.append(Charge=charges)


def distance(phi1,phi2,rapidity1,rapidity2):
		phi_dist = Components.angular_distance(phi1,phi2)
		rapidity_dist = rapidity1 - rapidity2
		return np.sqrt(phi_dist**2 + rapidity_dist**2)

def counting(PATH,which_jet):
		ew = Components.EventWise.from_file(PATH)
		M_c = np.zeros((40,40))
		M_n = np.zeros((40,40))
		M_p = np.zeros((40,40))

		b1 = np.where(ew.MCPID[0] == 5)[0][0]
		b2 = np.where(ew.MCPID[0] == -5)[0][0]
		b_pos=[ew.Phi[0][b1],ew.Rapidity[0][b1],ew.Phi[0][b2],ew.Rapidity[0][b2]]

		print(b_pos)
		for n_event in range(1000):
				ew.selected_index = n_event

				#finding bs shower
				index_leaf_b = JoinShower.finding_children(ew,5)
				index_leaf_d = JoinShower.finding_children(ew,-5)
 
				if which_jet=='bd':
						index_leaf = index_leaf_b + list(set(index_leaf_d)-set(index_leaf_b))
				elif which_jet == 'd':
						index_leaf = index_leaf_d
				elif which_jet == 'b':
						index_leaf = index_leaf_b
				#importing phi and rapidity, traslated to positive
				phi = ew.Phi + np.pi
				rapidity = ew.Rapidity + 2.5
				pt = ew.PT

				#importing charge
				charge = ew.Charge

				#create a 40*40 zeros matrix
				matrix_charged = np.zeros((40,40))
				matrix_neutral = np.zeros((40,40))
				matrix_pt = np.zeros((40,40))

				#steps
				phi_step = np.pi*2/40.
				rapidity_step = 5./40.

				for particle in index_leaf:
						col = int(phi[particle]/phi_step)
						row = int(rapidity[particle]/rapidity_step)

						if row<40 and row>0 and col<40 and col>0:
								if charge[particle] != 0:

										matrix_charged[row][col] += 1

								else:
										matrix_neutral[row][col] += pt[particle]

								matrix_pt[row][col] += pt[particle]
						
				M_c += matrix_charged
				M_n += matrix_neutral
				M_p += matrix_pt

		return np.rot90(M_c),np.rot90(M_n),np.rot90(M_p),b_pos




def loop_and_plot():
		fig,axs = plt.subplots(2,2,figsize=(10,10))
		ax = axs.flatten()

		PATH1 = '../data/gz.hepmc0.hepmc.awkd'
		PATH2 = '../data/gz_E.hepmc0.hepmc.awkd'
		PATH3 = '../data/hz.hepmc0.hepmc.awkd'
		PATH4 = '../data/hz_E.hepmc0.hepmc.awkd'

		PATH = [PATH1,PATH2,PATH3,PATH4]

		for i in range(4):

				m1,m2,m3,b_pos = counting(PATH[i],'bd')
		   
				'''
				if which_charge == 1:
						weights = np.reshape(m1,1600)
				elif which_charge ==0:
						weights = np.reshape(m2,1600)
				else:
						weights = np.reshape(m3,1600)

				'''
				#m3 = np.log2(m3+1)
				#weights=np.reshape(m3,1600)

				for ind,row in enumerate(m3):
					for ind2,col in enumerate(row):
						if col>0:
							checkpoint = np.log2(m3[ind][ind2]) 
							if checkpoint > 0:
								m3[ind][ind2] = checkpoint
							else:
								m3[ind][ind2] = 0

				weights = np.reshape(m3,1600)

				phi_rg = np.linspace(np.pi,-np.pi,40)
				rap_rg = np.linspace(-2.5,2.5,40)

				xs,ys=[],[]
				for p in range(40):
					for q in range(40):
						xs.append(rap_rg[q])
						ys.append(phi_rg[p])

				xs = np.array(xs)
				ys = np.array(ys)

				#plotting KDE
				xmin,xmax = xs.min(),xs.max()
				ymin,ymax = ys.min(),ys.max()
				X,Y = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
				positions = np.vstack([X.ravel(),Y.ravel()])
				values = np.vstack([xs,ys])
				#kernel = stats.gaussian_kde(values,0.1,weights=weights)
				#Z = np.reshape(kernel(positions).T,X.shape)

				zz = m3
				#img = ax[i].contourf(X,Y,Z,cmap='viridis')
				xx,yy = np.meshgrid(rap_rg,phi_rg)
				img = ax[i].imshow(zz)

				#img = ax[i].imshow(m3)
				#ax[i].scatter(b_pos[1],b_pos[0],marker='1',c='r')
				#ax[i].scatter(b_pos[3],b_pos[2],marker='2',c='r')
				#dist = distance(b_pos[0],b_pos[2],b_pos[1],b_pos[3])
				#ax[i].set_title('Dist: {}'.format(round(dist,4)))
				fig.colorbar(img,ax=ax[i])
				

		'''
		if which_charge ==1:
				plt.savefig(parent+'z_charge')
		elif which_charge ==0:
				plt.savefig(parent+'z_neutral')
		else:
				plt.savefig(parent+'z_pt')
		'''
		plt.savefig('prova3')
		plt.close()
		#plt.show()

'''
from pomegranate import GeneralMixtureModel,MultivariateGaussianDistribution,NormalDistribution,LogNormalDistribution
from matplotlib.colors import LogNorm
def try_gmm():
		fig,axs = plt.subplots(2,2,figsize=(10,10))
		ax = axs.flatten()

		PATH1 = '../data/hz.hepmc0.hepmc.awkd'
		PATH2 = '/scratch/gc2c20/data/restricted_kinematics/gz_E.hepmc1.hepmc.awkd'
		PATH3 = '/scratch/gc2c20/data/restricted_kinematics/hz_E.hepmc0.hepmc.awkd'
		PATH4 = '/scratch/gc2c20/data/restricted_kinematics/hz_E.hepmc1.hepmc.awkd'

		PATH = [PATH1,PATH2,PATH3,PATH4]

		for i in range(1):

				m1,m2,m3,b_pos = counting(PATH[i],'bd')
				#n1,n2,n3,b_pos = counting(PATH[i],'d')
 
				phi_rg = np.linspace(np.pi,-np.pi,40)
				rap_rg = np.linspace(-2.5,2.5,40)

				xx,yy = np.meshgrid(rap_rg,phi_rg)
				x_ = np.array(list(zip(xx.flatten(),yy.flatten())))
								
				#m3 = np.log2(m3+1)
				weight_m = np.reshape(m3,1600)
				#model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,1,x_,weights = weight_m)
				#model = NormalDistribution.from_samples(x_,weights=weight_m)
				#p2 = model.probability(x_).reshape(len(rap_rg),len(phi_rg))

				n3 = np.log2(n3+1)
				weight_n = np.reshape(n3,1600)

				model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,2,x_,weights = weight_n)
				q2 = model.probability(x_).reshape(len(rap_rg),len(phi_rg))

				#tot = p2+q2
				step=0.02
				#m = np.amax(p2)
				#n = np.amax(q2)
				#level_m = np.arange(0.0, m, step) + step
				#level_n = np.arange(0.0, n, step) + step
				#img = ax[i].contourf(xx,yy,p2,cmap='Blues',alpha=0.5)

				#img = ax[i].contourf(xx,yy,q2,level_n,cmap='Reds',alpha=0.5)
				#fig.colorbar(img,ax=ax[i])
				#ax[i].scatter(b_pos[1],b_pos[0],marker='1',c='r')
				#ax[i].scatter(b_pos[3],b_pos[2],marker='2',c='r')
 
				
		#plt.savefig('check')
		plt.close()
		#return(xx,m3)
		return x_, weight_m
'''		
