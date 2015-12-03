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

class ConstantOneFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        return [1.0]

    def length(self):
        return 1

class MaxHeightFeature(Feature):
    def __init__(self,degree):
        self.degree=degree

    def f(self,s,a):
        bitmap = a.get_prefinal_bitmap()
        H = bitmap.shape[0]
        minrow = H if bitmap.sum()==0 else bitmap.nonzero()[0].min()
        hpct = float(H - minrow)/float(H)

        return [hpct**i for i in range(1,self.degree+1)]

    def length(self):
        return self.degree


class DiscretizedMaxHeightFeature(Feature):
    def __init__(self,bins):
        self.bin_pairs = list(enumerate([float(v) for v in bins]))

    def f(self,s,a):
        def score(b):
            if b.sum()==0:
                return 0
            else:
                row_indices = np.nonzero(b)[0]
                return b.shape[0]-row_indices.min()
                
        def soft_discretize(score,bin_pairs):
            scores = [0] * len(bin_pairs)
            upper_bin_id,upval = next((i,v) for i,v in bin_pairs if raw_score<=v)
            lower_bin_id,lowval = next((i,v) for i,v in reversed(bin_pairs) if raw_score>=v)
            
            pctlower = (raw_score - lowval) / (upval - lowval)
            pctupper = 1.0 - pctlower
            
            scores[upper_bin_id] = pctupper
            scores[lower_bin_id] = pctlower
            return scores

        def hard_discretize(score,bin_pairs):
            scores = [0] * len(bin_pairs)
            upper_bin_id,upval = next((i,v) for i,v in bin_pairs if raw_score<=v)
            scores[upper_bin_id] = 1.0
            return scores
            

        bitmap = a.get_prefinal_bitmap()
        raw_score = score(bitmap)
        return hard_discretize(raw_score, self.bin_pairs)
        

    def length(self):
        return len(self.bin_pairs)

class DiscretizedMeanHeightFeature(Feature):
    def __init__(self,bins):
        self.bin_pairs = list(enumerate(bins))

    def f(self,s,a):
        def score(b):
            if b.sum()==0:
                return 0.0001
            else:
                row_indices = np.nonzero(b)[0]
                return b.shape[0]-row_indices.mean()

        def soft_discretize(score,bin_pairs):
            scores = [0] * len(bin_pairs)
            upper_bin_id,upval = next((i,v) for i,v in bin_pairs if raw_score<=v)
            lower_bin_id,lowval = next((i,v) for i,v in reversed(bin_pairs) if raw_score>=v)

            if upper_bin_id == lower_bin_id:
                scores[upper_bin_id] = 1.0
                return scores
            else:
                pctlower = (raw_score - lowval) / (upval - lowval)
                pctupper = 1.0 - pctlower
                
                scores[upper_bin_id] = pctupper
                scores[lower_bin_id] = pctlower
                return scores

        scores = [0] * len(self.bin_pairs)
        bitmap = a.get_prefinal_bitmap()
        raw_score = score(bitmap)
        return soft_discretize(raw_score, self.bin_pairs)

    def length(self):
        return len(self.bin_pairs)

class MeanHeightFeature(Feature):
    def __init__(self,degree):
        self.degree = degree

    def f(self,s,a):
        def score(b):
            if b.sum()==0:
                return 0
            else:
                row_indices = np.nonzero(b)[0]
                return b.shape[0]-row_indices.mean()

        bitmap = a.get_prefinal_bitmap()
        H = float(bitmap.shape[0])
        scr = float(score(bitmap))/float(H)
        return [scr**i for i in range(1,self.degree+1)]

    def length(self):
        return self.degree

class MeanHeightSquaredFeature(Feature):
    def __init__(self):
        pass

    def f(self,s,a):
        def score(b):
            if b.sum()==0:
                return 0
            else:
                row_indices = np.nonzero(b)[0]
                return b.shape[0]-row_indices.mean()

        bitmap = a.get_prefinal_bitmap()
        return [score(bitmap)**2]

    def length(self):
        return 1

class RowLengthHistogramFeature(Feature):
    def __init__(self,width):
        self.width=width

    def f(self,s,a):
        bitmap = a.get_prefinal_bitmap()
        rowlengths = bitmap.sum(axis=1).astype(np.int_)
        hist = [0] * (self.width)
        for i in rowlengths:
            hist[i] += 1
        return hist

    def length(self):
        return self.width

class RowStateHistogramFeature(Feature):
    def __init__(self,width,degree):
        self.width=width
        self.degree=degree

    def f(self,s,a):
        bitmap = a.get_prefinal_bitmap().copy()

        # Set blocked squares to have cell value 2
        for c in range(bitmap.shape[1]):
            flag=False
            for r in range(bitmap.shape[0]):
                if bitmap[r,c]==1:
                    flag=True
                elif flag:
                    bitmap[r,c] = 2

        zeros = (bitmap==0).sum(axis=1, keepdims=True)
        ones = (bitmap==1).sum(axis=1, keepdims=True)
        twos = (bitmap==2).sum(axis=1, keepdims=True)
        M = np.hstack((zeros,ones,twos))

        
        H = bitmap.shape[0]
        C = bitmap.shape[1]
        assert not np.any(M>C)
        assert np.all(M.sum(axis=1)==C)
                          
        hist = [0] * ((C+1) * (C+1))
        for row in M:
            index = row[0]*(C+1) + row[1]
            hist[index] += 1.0/H

        return list(itertools.chain(*([i**k for i in hist] for k in range(1,self.degree+1))))

    def length(self):
        return (self.width+1)*(self.width+1)*self.degree

class ColHeightHistogramFeature(Feature):
    def __init__(self,height):
        self.height=height

    def f(self,s,a):
        bitmap = a.get_prefinal_bitmap()
        H = bitmap.shape[0]
        assert H == self.height
        colheights = bitmap.sum(axis=0).astype(np.int_)
        hist = [0] * (H+1)
        for i in colheights:
            hist[i] += 1
        return hist

    def length(self):
        return self.height+1

class TrappedSquaresFeature(Feature):
    def __init__(self,degree):
        self.degree=degree

    def f(self,s,a):
        def col_num_trapped(col):
            nz = np.nonzero(col)[0]
            if nz.size == 0: 
                return 0
            top = nz.min()
            h = col.size - top
            nnz = nz.size
            return h-nnz

        def bitmap_num_trapped(b):
            return sum(col_num_trapped(b[:,c]) for c in range(b.shape[1]))

        bitmap = a.get_final_bitmap()
        n = bitmap_num_trapped(bitmap)
        pct = float(n) / float(bitmap.size)
        return [pct ** k for k in range(1,self.degree+1)]

    def length(self):
        return self.degree


class SquareTypesFeature(Feature):
    def __init__(self,degree):
        self.degree=degree

    def f(self,s,a):
        def col_num_trapped(col):
            nz = np.nonzero(col)[0]
            if nz.size == 0: 
                return 0
            top = nz.min()
            h = col.size - top
            nnz = nz.size
            return h-nnz

        def bitmap_num_trapped(b):
            return sum(col_num_trapped(b[:,c]) for c in range(b.shape[1]))

        bitmap = a.get_final_bitmap()
        n = bitmap_num_trapped(bitmap)
        pct = float(n) / float(bitmap.size)
        return [pct ** k for k in range(1,self.degree+1)]

    def length(self):
        return self.degree

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
        bitmap = a.get_prefinal_bitmap()

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

    def names_and_lengths(self):
        return list(zip(self.names, [f.length() for f in self.functions]))

