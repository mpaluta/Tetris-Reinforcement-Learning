import numpy as np
import sys
import itertools
import environment


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


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

class MaxHeightFeature(Feature):
    def __init__(self):
        pass
    def f(self,s,a):
        bitmap = s.arena.bitmap
        H = bitmap.shape[1]
        minrow = bitmap.nonzero()[0].min()
        maxheight = H - minrow
        return [maxheight]

    def length(self):
        return 1

class FeatureFunctionVector(object):
    def __init__(self, config):
        self.functions = []
        for feat_name, feat_args in config.iteritems():
            try:
                self.functions.add(str_to_class(feat_name)(*feat_args))
            except AttributeError:
                raise Exception("Cannot find feature with name '{}'".format(feat_name))

    def f(self, s, a):
        fvecs = [f.f(s,a) for f in self.functions]
        return np.array(itertools.chain(*fvecs))

    def length(self):
        return sum(f.length() for f in self.functions)
            
