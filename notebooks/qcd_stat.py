import numpy as np
from jet_tools import Components
from statistics import mean
import matplotlib.pyplot as plt

def number_of_GtoGG(PATH):
	ew = Components.EventWise.from_file(PATH)
	count=[]
	mcpid=[1,2,3,4,5,6,-1,-2,-3,-4,-5,-6]
	for i in range(1000):
		ew.selected_index=i
		g = np.where(ew.MCPID == 21)[0]
		check=[]
		for l in g:
			if len(ew.Children[l])==2:
				check.append(l)

		c=0
		for l in check:
			child1 = ew.Children[l][0]
			child2 = ew.Children[l][0]
			child1_id = ew.MCPID[child1]
			child2_id = ew.MCPID[child2]
			child1_parent = ew.Parents[child1]
			child2_parent = ew.Parents[child2]
			'''
			if child1_id==21 and child2_id==21 and len(child1_parent)==1 and len(child2_parent)==1:
				c+=1
			'''
			if len(child1_parent)==1 and len(child2_parent)==1:
				if any(pid == child1_id for pid in mcpid) or any(pid==child2_id for pid in mcpid):
					c+=1				

		count.append(c)

	return np.array(count)

def plot_bars(PATH1,PATH2):
	x = np.linspace(0,1000,1000)
	ct = number_of_GtoGG(PATH1)
	ct_enhance = number_of_GtoGG(PATH2)

	ct_mean = mean(ct)
	ct_enhance_mean = mean(ct_enhance)

	fig,ax = plt.subplots(2,1,figsize=(10,10))
	ax[0].bar(x,ct)
	ax[0].hlines(ct_mean,min(x),max(x),colors='r',linestyles='dashed')
	ax[0].set_ylim(0,200)
	
	ax[1].bar(x,ct_enhance)
	ax[1].hlines(ct_enhance_mean,min(x),max(x),colors='r',linestyles='dashed')
	ax[1].set_ylim(0,200)


	plt.savefig('hist')
	plt.close()

	return 
