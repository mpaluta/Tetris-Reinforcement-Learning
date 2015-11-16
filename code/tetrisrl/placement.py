import environment
import numpy as np

class Node(object):
    s=None
    bp=None
    t=None

    def __init__(self,s,bp):
        

class BitmapHash(object):
    def __init__(self):
        self._dict={}

    def add(self, bitmap, trajectory):
        key = tuple(bitmap.flatten().tolist())
        if key in self._dict:
            b,t = self._dict[key]
            if len(trajectory) < len(t):
                self._dict[key] = (bitmap,trajectory)
        else:
            self._dict[key] = (bitmap,trajectory)
            
    def get_bitmap_trajectory_pairs(self):
        return self._dict.values()
    

class PlacementEnumerator(object):
    def __init__(self):
        pass

    def _get_successor_bitmaps(self, e, s, bh, t):
        results = []
        for a in environment.all_actions:
            sprime, _ = e.next_state_and_reward(s,a)
            if sprime.arena.bitmap != s.arena.bitmap:
                # new bitmap
                bh.add(bitmap,t+[a])
            else:
                self.get_successor_bitmaps(e,sprime,bh,t+[a])
    
    def get_successor_bitmaps(self, e, s):
        bh = BitmapHash()
        t = []
        self._get_successor_bitmaps(e, s, bh, t)
        return bh.get_bitmap_trajectory_pairs()

