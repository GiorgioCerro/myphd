import numpy as np
from jet_tools import Components

class JoinShower:
    def finding_children(self,ew,p_idw):
        start_jet = np.where(ew.MCPID == p_idw)[0][-1]

        first_children = ew.Children[start_jet].tolist()
        leaf,children = [],[]


        while len(first_children)>0: 
            children=[]
            for child in first_children:
                children_help = ew.Children[child].tolist()
                for child_help in children_help:
                    if len(ew.Children[child_help].tolist()) < 1:
                        if child_help not in leaf:
                            leaf.append(child_help)
                    else:
                        if child_help not in children: 
                            children.append(child_help)
            first_children = children
    
        return leaf
