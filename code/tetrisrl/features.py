import numpy as np
import sys
import itertools
import environment


def _str_to_class(_str):
    return getattr(sys.modules[__name__], _str)


class Feature(object):
    def __init__(self):
        pass

    def f(self,s,a):
        raise Exception("Cannot instantiate abstract Feature class")

class MinorActionCounterFeature(Feature):
    def __init__(self):
        self.num_actions = len(environment.all_actions)

    def f(self,s,a):
        v = [0] * self.num_actions
        for i in a.minor_actions:
            assert i < self.num_actions and i>0
            v[i] += 1
        return v

    def length(self):
        return self.num_actions

class ConstantOneFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        return [1.0]

    def length(self):
        return 1

class ShortestRowCompletionLengthFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_prefinal_bitmap()
        W = bitmap.shape[1]
        ws = bitmap.sum(axis=1)
        maxw = ws.max()
        return [float(W-maxw)]

    def length(self):
        return 1

class MaxHeightChangeFeature(Feature):
    def __init__(self):
        pass

    def h(self,b):
        H = b.shape[0]
        minrow = H if b.sum()==0 else b.nonzero()[0].min()
        maxheight = H - minrow
        return maxheight

    def f(self,s,a):
        final_h = self.h(a.get_final_bitmap())
        initial_h = self.h(s.arena.bitmap)

        return [final_h - initial_h]


    def length(self):
        return 1


class MaxHeightFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_final_bitmap()
        H = bitmap.shape[0]
        minrow = H if bitmap.sum()==0 else bitmap.nonzero()[0].min()
        maxheight = H - minrow
        return [maxheight]

    def length(self):
        return 1

class TrappedSquaresFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_final_bitmap()
        H = bitmap.shape[0]
        W = bitmap.shape[1]
        badcount=0
        for c in range(W):
            count = 0
            for r in range(H-1,-1,-1):
                if bitmap[r,c]==0:
                    count += 1
                else:
                    badcount += count
        return [badcount]

    def length(self):
        return 1

class MaxHeightSquaredFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_final_bitmap()
        H = bitmap.shape[0]
        minrow = H if bitmap.sum()==0 else bitmap.nonzero()[0].min()
        maxheight = H - minrow
        return [maxheight**2]

    def length(self):
        return 1

class MinHeightFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_final_bitmap()
        H = bitmap.shape[0]
        W = bitmap.shape[1]
        maxrow = 0
        for col in range(W):
            for row in range(H):
                if row > maxrow:
                    maxrow = row
                if bitmap[row,col] == 1:
                    break
                if row == H-1:
                    maxrow = H
        minheight = H - maxrow
        return [minheight]

    def length(self):
        return 1

class HeightDiffFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_final_bitmap()
        H = bitmap.shape[0]
        W = bitmap.shape[1]
        minrow = bitmap.nonzero()[0].min() # minimum row index of a one
        maxheight = H - minrow
        maxrow = 0
        for col in range(W):
            for row in range(H):
                if row > maxrow:
                    maxrow = row
                if bitmap[row,col] == 1:
                    break
                if row == H-1:
                    maxrow = H
        minheight = H - maxrow
        diff = maxheight - minheight
        return [diff]

    def length(self):
        return 1

class HeightVarFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_final_bitmap()
        H = bitmap.shape[0]
        W = bitmap.shape[1]
        heights = np.zeros(W)
        for col in range(W):
            maxrow = 0
            for row in range(H):
                if row > maxrow:
                    maxrow = row
                if bitmap[row,col] == 1:
                    break
                if row == H-1:
                    maxrow = H
            heights[col] = H - maxrow
        heightvar = np.var(heights)
        return [heightvar]

    def length(self):
        return 1

class RowTransitionsFeature(Feature): 
    # sum of horizontal occupied/unoccupied transitions on board
    # left and right side of board count as occupied

    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_final_bitmap()
        H = bitmap.shape[0]
        W = bitmap.shape[1]
        count = 0
        last = 1
        for row in range(H):
            last = 1
            for col in range(W):
                if bitmap[row,col] != last:
                    last = bitmap[row,col]
                    count += 1
                if col == W-1 and bitmap[row,col] == 0:
                    count += 1
        return [count]

    def length(self):
        return 1

class ColTransitionsFeature(Feature):
    # sum of vertical occupied/unoccupied transitions on board
    # above board counts as unoccupied; below board counts as occupied

    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_final_bitmap()
        H = bitmap.shape[0]
        W = bitmap.shape[1]
        count = 0
        last = 0
        for col in range(W):
            last = 0
            for row in range(H):
                if bitmap[row,col] != last:
                    last = bitmap[row,col]
                    count += 1
                if row == H-1 and bitmap[row,col] == 0:
                    count += 1
        return [count]

    def length(self):
        return 1

class CompactnessFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        bitmap = a.get_final_bitmap()

        if bitmap.sum() == 0:
            return [1.0]
        else:
            H = bitmap.shape[0]
            W = bitmap.shape[1]
            heights = np.zeros(W)
            for col in range(W):
                maxrow = 0
                for row in range(H):
                    if row > maxrow:
                        maxrow = row
                    if bitmap[row,col] == 1:
                        break
                    if row == H-1:
                        maxrow = H
                heights[col] = H - maxrow
            meanheight = np.mean(heights)
            blocks = np.sum(bitmap)
            compactness = blocks/(meanheight*W)
            return [compactness]

    def length(self):
        return 1

# Further ideas for features:

# Holes - unoccupied cells with at least one occupied cell above
# Connected Holes - vertically connected cells count as single hole 
# Removed lines - number of lines cleared in last step to get to current board
# Maximum well depth - deepest well of width one
# Sum of all wells
# Landing height - height at which last tetramino has been placed

class FeatureFunctionVector(object):
    def __init__(self, config):
        self.functions = []
        self.names = []
        for feat_name, feat_args in sorted(config.iteritems()):
            try:
                cls = _str_to_class(feat_name)
                self.functions.append(cls(**feat_args))
                self.names.append(feat_name)
            except AttributeError:
                raise Exception("Cannot find feature with name '{}'".format(feat_name))

    def f(self, s, a):
        fvecs = [f.f(s,a) for f in self.functions]
        return np.array(list(itertools.chain(*fvecs)))

    def length(self):
        return sum(f.length() for f in self.functions)
            
