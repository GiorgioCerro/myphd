import numpy as np
from jet_tools import Components
from pyjet import cluster

class AntiKt():
	def __init__(self,PATH,R):
		self.PATH = PATH
		self.R = R

	def get_data(self,index):
		ew = Components.EventWise.from_file(self.PATH)
		ew.selected_index = index
		px,py,pz = ew.JetInputs_Px,ew.JetInputs_Py,ew.JetInputs_Pz
		E = ew.JetInputs_Energy

		event = np.zeros(len(E),dtype={'names':('E','px','py','pz'),
					'formats':('f8','f8','f8','f8')})

		event['px'],event['py'],event['pz'] = px,py,pz
		event['E'] = E

		return event

	def clustering(self,event):
		sequence = cluster(event, self.R, p=-1, ep=True)
		return sequence.inclusive_jets()
	

	def filters(self,clusters):	
		count = 0
		for j in jets:
			if j.pt>30 and len(j.constituents_array())>2:
				count+=1	

		return count


	def jet_multiplicity(self):
		mult = []
		for i in range(10):
			event = self.get_data(i)
			jets = self.clustering(event)
			mult.append(self.filters(jets))
		
		return mult
