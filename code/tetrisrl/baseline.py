import environment
from placement import PlacementEnumerator
import numpy as np
import logging


class LowestCenterOfGravityAgent(object):
    penum=None
    queued_actions=None
    e=None
    def __init__(self,e):
        self.penum=PlacementEnumerator(e)
        self.queued_actions=[]

    def act(self,s,debug_mode=False):
        def score(b):
            row_indices = np.nonzero(b)[0]
            return -row_indices.mean()

        if len(self.queued_actions)>0:
            a = self.queued_actions[0]
            self.queued_actions = self.queued_actions[1:]
            return (a,None)
        else:
            logging.info("DELTA: 0")
            a_s_p = self.penum._get_actionseq_finalstate_pairs(s)
            assert(len(a_s_p)>0)
            amin,smin,pmin = min(a_s_p, key=lambda x:score(x[2]))
            #assert pmin.sum()==0 or np.nonzero(pmin)[0].min() == pmin.shape[0]-1

            self.queued_actions = amin[1:]
            return (amin[0], None)
                        
    def observe_sars_tuple(self,s,a,r,sprime,pfbm=None):
        logging.info("REWARD: {}".format(r))
        pass
        
    def save_model(self,fn):
        pass
