import numpy as np
from jet_tools import Components,TrueTag
from pyjet import cluster
import matplotlib.pyplot as plt
from join_shower import JoinShower
from scipy.optimize import curve_fit

class AntiKt():
	def __init__(self,R):
		self.R = R

	def get_data(self,ew):
		em_shower = JoinShower.finding_children(ew,23)
		inputs_idx = ew.JetInputs_SourceIdx
		leaf = list(set(inputs_idx) - set(em_shower))
		px,py,pz = ew.Px[leaf],ew.Py[leaf],ew.Pz[leaf]
		E = ew.Energy[leaf]

		event = np.zeros(len(E),dtype={'names':('E','px','py','pz'),
					'formats':('f8','f8','f8','f8')})

		event['px'],event['py'],event['pz'] = px,py,pz
		event['E'] = E

		return event

	def clustering(self,event,param):
		sequence = cluster(event, R = param, p=-1, ep=True)
		return sequence.inclusive_jets()
	

	def filters(self,clusters,pt_cut):	
		count = 0
		pass_jets=[]
		for j in clusters:
			if j.pt>pt_cut and len(j.constituents_array())>2:
				count+=1	
				pass_jets.append(j)

		return count,np.array(pass_jets)

	def tag_b(self,ew,jets):
		b1 = np.where(ew.MCPID==5)[0][-1]
		b2 = np.where(ew.MCPID==-5)[0][-1]

		tags = [[ew.Rapidity[b1],ew.Phi[b1]],[ew.Rapidity[b2],ew.Phi[b2]]]
		tags = np.array(tags)

		j=[]
		for i in jets:
			j.append([i.eta,i.phi])
		j = np.array(j)

		bs = TrueTag.allocate(j[:,1],j[:,0],tags[:,1],tags[:,0],0.8**2)
		j_b = []
		#print(bs[0]==bs[1])
		if bs[0]==bs[1]:
			j_b.append(jets[bs[0]])
			'''
			j = list(j)
			j.pop(bs[0])
			'''
		else:
			j_b.append(jets[bs[0]])
			j_b.append(jets[bs[1]])
			'''
			j = list(j)
			#print(len(j),'-',bs[0],'-',bs[1])
			if len(j)>2:
				j.pop(bs[0])
				j.pop(bs[1]-1)
			elif len(j)==2:
				j.pop(0)
				j.pop(0)
			else:
				j.pop(bs[0])
			'''
		return j_b #,j

	def get_jets(self,PATH,index):
		ew = Components.EventWise.from_file(PATH)
		ew.selected_index = index
		event = self.get_data(ew)
		jets = self.clustering(event,self.R)
		count,pass_jets = self.filters(jets,20)
		return pass_jets

	def jet_multiplicity(self,PATH,r):
		mult = []
		ew = Components.EventWise.from_file(PATH)
		
		j_b,j_others = [],[]
		for i in range(2000):
			#print(i)
			ew.selected_index = i
			event = self.get_data(ew)
			jets = self.clustering(event,r)
			count,pass_jets = self.filters(jets,50)
			mult.append(count)
			'''
			jb,jo = self.tag_b(ew,pass_jets)
			j_others.append(jo)
			j_b.append(jb)
			'''	

		multiplicity = np.zeros(9)
		for jet in mult:
			multiplicity[jet]+=1

		return multiplicity #,j_others,j_b

	def jet_submultiplicity(self,PATH):
		mult=[]
		ew = Components.EventWise.from_file(PATH)
		
		for i in range(2000):
			ew.selected_index = i
			event = self.get_data(ew)
			jets = self.clustering(event,self.R)
			count,pass_jets = self.filters(jets,30)

			j_b = self.tag_b(ew,pass_jets)
			children=[]

			if len(j_b) == 1:
				children.append(j_b[0].constituents_array(ep=True))
				children = [item for sublist in children for item in sublist]
			else:
				children.append(j_b[0].constituents_array(ep=True))
				children.append(j_b[1].constituents_array(ep=True))
				children = [item for sublist in children for item in sublist]
	
			px,py,pz,E = [],[],[],[]		
			for child in children:
				E.append(child[0])
				px.append(child[1])
				py.append(child[2])
				pz.append(child[3])
			#print(i,'---',len(children),'---',len(px))
			px = np.array(px)
			py = np.array(py)
			pz = np.array(pz)
			E = np.array(E)

			event = np.zeros(len(E),dtype={'names':('E','px','py','pz'),
					'formats':('f8','f8','f8','f8')})

			event['px'],event['py'],event['pz'] = px,py,pz
			event['E'] = E

			jets = self.clustering(event,0.4)
			count,pass_jets2 = self.filters(jets,20)
			
			mult.append(count)


		multiplicity = np.zeros(9)
		for jet in mult:
			multiplicity[jet]+=1

		return multiplicity

	
	
	def get_plot(self,sub_mul=False):
		PATH_G = '../data/gz.hepmc0.hepmc.parquet'
		PATH_H = '../data/hz.hepmc0.hepmc.parquet'
		PATH_GE = '../data/gz_E.hepmc0.hepmc.awkd'
		PATH_HE = '../data/hz_E.hepmc0.hepmc.awkd'

		x_data = np.arange(1,10)
		if sub_mul==True:
			y_data_G = self.jet_submultiplicity(PATH_G)
			y_data_H = self.jet_submultiplicity(PATH_H)
			y_data_GE = self.jet_submultiplicity(PATH_GE)
			y_data_HE = self.jet_submultiplicity(PATH_HE)

		else:
			y_data_G = self.jet_multiplicity(PATH_G,0.4)
			y_data_H = self.jet_multiplicity(PATH_H,0.4)
			y_data_G1 = self.jet_multiplicity(PATH_G,0.8)
			y_data_H1 = self.jet_multiplicity(PATH_H,0.8)
	
			y_data_G2 = self.jet_multiplicity(PATH_G,1.0)
			y_data_H2 = self.jet_multiplicity(PATH_H,1.0)

	
		'''	

		fig, ax = plt.subplots(2,2,figsize=(5,5))
		#ax.flatten()

		label_a = 'gluon'
		label_b = 'higgs'
		label_c = 'gluon_NOggg'
		label_d = 'higgs_NOggg'
		ax[0][0].step(x_data,y_data_G,'k',where='mid',linewidth=1,c='r',label=label_a)
		ax[0][0].step(x_data,y_data_H,'k',where='mid',linewidth=1,c='b',label=label_b)
		ax[0][0].legend(loc='upper right')

		ax[0][1].step(x_data,y_data_GE,'k',where='mid',linewidth=1,c='brown',label=label_c)
		ax[0][1].step(x_data,y_data_HE,'k',where='mid',linewidth=1,c='g',label=label_d)
		ax[0][1].legend(loc='upper right')

		ax[1][0].step(x_data,y_data_G,'k',where='mid',linewidth=1,c='r',label=label_a)
		ax[1][0].step(x_data,y_data_GE,'k',where='mid',linewidth=1,c='brown',label=label_c)
		ax[1][0].legend(loc='upper right')

		ax[1][1].step(x_data,y_data_H,'k',where='mid',linewidth=1,c='b',label=label_b)
		ax[1][1].step(x_data,y_data_HE,'k',where='mid',linewidth=1,c='g',label=label_d)
		ax[1][1].legend(loc='upper right')

		'''
		fig,ax = plt.subplots(1,3,figsize=(5,5))
		label_a = 'gluon'
		label_b = 'higgs'
		ax[0].step(x_data,y_data_G,'k',where='mid',linewidth=1,c='r',label=label_a)
		ax[0].step(x_data,y_data_H,'k',where='mid',linewidth=1,c='b',label=label_b)
		ax[0].legend(loc='upper right')
		ax[1].step(x_data,y_data_G1,'k',where='mid',linewidth=1,c='r',label=label_a)
		ax[1].step(x_data,y_data_H1,'k',where='mid',linewidth=1,c='b',label=label_b)
		ax[1].legend(loc='upper right')
		ax[2].step(x_data,y_data_G2,'k',where='mid',linewidth=1,c='r',label=label_a)
		ax[2].step(x_data,y_data_H2,'k',where='mid',linewidth=1,c='b',label=label_b)
		ax[2].legend(loc='upper right')

		fig.suptitle('Jet multiplicity: 0.4, 0.8, 1.0',fontsize=24)
		plt.show()





	def get_charge(self,PATH):
		ew = Components.EventWise.from_file(PATH)
		charge_jet = []
		for i in range(2000):
			ew.selected_index = i
			event = self.get_data(ew)
			jets = self.clustering(event,self.R)
			count,pass_jets = self.filters(jets,30)

			j_b = self.tag_b(ew,pass_jets)
			children=[]

			if len(j_b) == 1:
				children.append(j_b[0].constituents_array(ep=True))
				children = [item for sublist in children for item in sublist]
			else:
				children.append(j_b[0].constituents_array(ep=True))
				children.append(j_b[1].constituents_array(ep=True))
				children = [item for sublist in children for item in sublist]
	
			indeces = []
			px,py,pt = [],[],[]
			chi = 0.20
			for child in children:
				energy = child[0]
				px.append(child[1])
				py.append(child[2])
				pt.append(np.sqrt(child[1]**2 + child[2]**2))
				ind = np.where(ew.Energy == energy)[0]
				if len(ind) >1:
					ind = np.array([ind[-1]])
				indeces.append(ind)	

			#print(i,'---',len(children),'---',len(px))
			pt_jet = np.sqrt(sum(px)**2 + sum(py)**2)
			indeces = np.array(indeces)

			charge = ew.Charge[indeces]	
			charge_constituent = 0
			for cha in range(len(charge)):
				charge_constituent += charge[cha]*(pt[cha]**chi)
			
			#print(f'{i}---{charge_constituent}---{pt_jet}')
			charge_jet.append(charge_constituent/(pt_jet**chi))
			#print(charge_jet)


		return np.array(charge_jet)


	def gaussian(self,x,a,mean,sigma_sqr):
		return a*np.exp(-(x-mean)**2/(2*sigma_sqr))

	def error_prop(self,value,error):
		#this works if you want the sqrt value
		V = np.sqrt(value)
		E = 0.5 * error * V / value
		return V,E

	def get_jet_charge(self):
		PATH_G = '../data/gz.hepmc0.hepmc.awkd'
		PATH_H = '../data/hz.hepmc0.hepmc.awkd'
		PATH_GE = '../data/gz_E.hepmc0.hepmc.awkd'
		PATH_HE = '../data/hz_E.hepmc0.hepmc.awkd'
		PATH_GQ = '../data/gz_EQ.hepmc0.hepmc.awkd'
		PATH_HQ = '../data/hz_EQ.hepmc0.hepmc.awkd'
		
		charge_G = self.get_charge(PATH_G)
		charge_H = self.get_charge(PATH_H)

		charge_GE = self.get_charge(PATH_GE)
		charge_HE = self.get_charge(PATH_HE)
		charge_GQ = self.get_charge(PATH_GQ)
		charge_HQ = self.get_charge(PATH_HQ)

		

		bins = np.linspace(-3.5,3.5,100)
		
		fig,ax = plt.subplots(3,2,figsize=(5,5))
		G = ax[0][0].hist(charge_G,bins=bins,fc=(0,0,1,0.5))
		H = ax[0][1].hist(charge_H,bins=bins,fc=(1,0,0,0.5))


		Gx,Gy = G[1][:-1],G[0]
		G_popt,G_pcov = curve_fit(self.gaussian,Gx,Gy,p0=[1,0,0.2])
		G_mean, G_var = G_popt[1],G_popt[2]
		G_errors = np.sqrt(np.diag(G_pcov))
		G_sigma,G_sigma_err = self.error_prop(G_var,G_errors[2])
		ax[0][0].plot(Gx,self.gaussian(Gx,*G_popt),'-',color='blue',
			label=f'Gluon \n \u03BC={G_mean:.2f} \n \u03C3={G_sigma:.2f} \u00B1 {G_sigma_err:.2f}')

		Hx,Hy = H[1][:-1],H[0]
		H_popt,H_pcov = curve_fit(self.gaussian,Hx,Hy,p0=[1,0,0.2])
		H_mean, H_var = H_popt[1],H_popt[2]
		H_errors = np.sqrt(np.diag(H_pcov))
		H_sigma,H_sigma_err = self.error_prop(H_var,H_errors[2])
		ax[0][1].plot(Hx,self.gaussian(Hx,*H_popt),'-',color='red',
			label=f'Higgs \n \u03BC={H_mean:.2f} \n \u03C3={H_sigma:.2f} \u00B1 {H_sigma_err:.2f}')



		GE = ax[1][0].hist(charge_GE,bins=bins,fc=(0,0,1,0.5))
		HE = ax[1][1].hist(charge_HE,bins=bins,fc=(1,0,0,0.5))


		GEx,GEy = GE[1][:-1],GE[0]
		GE_popt,GE_pcov = curve_fit(self.gaussian,GEx,GEy,p0=[1,0,0.2])
		GE_mean, GE_var = GE_popt[1],GE_popt[2]
		GE_errors = np.sqrt(np.diag(GE_pcov))
		GE_sigma,GE_sigma_err = self.error_prop(GE_var,GE_errors[2])
		ax[1][0].plot(GEx,self.gaussian(GEx,*GE_popt),'-',color='blue',
			label=f'Gluon_NOggg \n \u03BC={GE_mean:.2f} \n \u03C3={GE_sigma:.2f} \u00B1 {GE_sigma_err:.2f}')

		HEx,HEy = HE[1][:-1],HE[0]
		HE_popt,HE_pcov = curve_fit(self.gaussian,HEx,HEy,p0=[1,0,0.2])
		HE_mean, HE_var = HE_popt[1],HE_popt[2]
		HE_errors = np.sqrt(np.diag(HE_pcov))
		HE_sigma,HE_sigma_err = self.error_prop(HE_var,HE_errors[2])
		ax[1][1].plot(HEx,self.gaussian(HEx,*HE_popt),'-',color='red',
			label=f'Higgs_NOggg \n \u03BC={HE_mean:.2f} \n \u03C3={HE_sigma:.2f} \u00B1 {HE_sigma_err:.2f}')



		GQ = ax[2][0].hist(charge_GQ,bins=bins,fc=(0,0,1,0.5))
		HQ = ax[2][1].hist(charge_HQ,bins=bins,fc=(1,0,0,0.5))


		GQx,GQy = GQ[1][:-1],GQ[0]
		GQ_popt,GQ_pcov = curve_fit(self.gaussian,GQx,GQy,p0=[1,0,0.2])
		GQ_mean, GQ_var = GQ_popt[1],GQ_popt[2]
		GQ_errors = np.sqrt(np.diag(GQ_pcov))
		GQ_sigma,GQ_sigma_err = self.error_prop(GQ_var,GQ_errors[2])
		ax[2][0].plot(GQx,self.gaussian(GQx,*GQ_popt),'-',color='blue',
			label=f'Gluon_NOgqq \n \u03BC={GQ_mean:.2f} \n \u03C3={GQ_sigma:.2f} \u00B1 {GQ_sigma_err:.2f}')

		HQx,HQy = HQ[1][:-1],HQ[0]
		HQ_popt,HQ_pcov = curve_fit(self.gaussian,HQx,HQy,p0=[1,0,0.2])
		HQ_mean, HQ_var = HQ_popt[1],HQ_popt[2]
		HQ_errors = np.sqrt(np.diag(HQ_pcov))
		HQ_sigma,HQ_sigma_err = self.error_prop(HQ_var,HQ_errors[2])
		ax[2][1].plot(HQx,self.gaussian(HQx,*HQ_popt),'-',color='red',
			label=f'Higgs_NOgqq \n \u03BC={HQ_mean:.2f} \n \u03C3={HQ_sigma:.2f} \u00B1 {HQ_sigma_err:.2f}')


		
		ax[0][0].legend(loc='upper right')
		ax[0][1].legend(loc='upper right')
		ax[1][0].legend(loc='upper right')
		ax[1][1].legend(loc='upper right')
		ax[2][0].legend(loc='upper right')
		ax[2][1].legend(loc='upper right')
		fig.suptitle('jet_charge with \u03BA =0.2',fontsize=24)
		plt.show()




