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
            bmin = None
            tmin = None
            scoremin=None
            #print "bts={}".format(bts)
            assert(len(bts)>0)
            for sprime,actions,r in bts:
                if scoremin is None or score(sprime.arena.bitmap)<scoremin:
                    scoremin,bmin,tmin = (score(sprime.arena.bitmap), sprime.arena.bitmap, actions)
            #print "Tmin={}".format(tmin)
            self.queued_actions = tmin[1:]
            return tmin[0]
                        
