import environment
from placement import PlacementEnumerator
import numpy as np


class LowestCenterOfGravityAgent(object):
    penum=None
    queued_actions=None
    e=None
    def __init__(self,e):
        self.penum=PlacementEnumerator()
        self.queued_actions=[]
        self.e = e

    def act(self,s):
        def score(b):
            row_indices = np.nonzero(b)[0]
            return -row_indices.mean()

        if len(self.queued_actions)>0:
            a = self.queued_actions[0]
            self.queued_actions = self.queued_actions[1:]
            return a
        else:
            bts = self.penum.get_successor_bitmaps(self.e, s)
            assert(len(bts)>0)
            smin,amin,rmin = min(bts, key=lambda x:score(x[0].arena.bitmap))
            self.queued_actions = amin[1:]
            return amin[0]
                        
