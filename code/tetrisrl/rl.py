import environment
from placement import PlacementEnumerator, PlacementAction
from features import FeatureFunctionVector
from baseline import LowestCenterOfGravityAgent
import collections
import numpy as np
import sys
import random
import logging


def str_to_class(_str):
    return getattr(sys.modules[__name__], _str)

def json_descr_to_class(obj, **kwargs):
    type_str = obj["type"]
    params = obj["params"]
    merged_params = params.copy()
    merged_params.update(kwargs)
    return str_to_class(type_str)(**merged_params)

class QModel(object):
    def __init__(self, config):
        self.features = FeatureFunctionVector(config["features"])
        self.initialize_weights(config["initial_weights"])

    def initialize_weights(self, config):
        n = self.features.length()
        if config["type"] == "uniform_random":
            self.weights = (np.random.random((n,)) * 2.0)-1.0
        elif config["type"] == "zero":
            self.weights = np.zeros((n,))
        elif config["type"] == "explicit":
            name_to_weights = config["values"]
            names_and_lengths = self.features.names_and_lengths()
            assert set(name_to_weights.keys()) == set(list(n for n,l in names_and_lengths))
            weights_list = []
            for name,length in names_and_lengths:
                ws = name_to_weights[name]
                assert length == len(ws), "Length {} does not equal number of weights specified {} for weight named {}".format(length, len(ws), name)
                weights_list.extend(ws)
            self.weights = np.array(weights_list, dtype=np.float_)
        else:
            raise Exception("Unexpected type: {}. Possibilities are: {}".format(config["type"], ["uniform_random","zero","explicit"]))

    def q(self, s, a):
        return self.weights.dot(self.features.f(s,a))

    def beta(self, s, a):
        return self.features.f(s,a)

    def update_weights(self, delta):
        self.weights += delta

    def save_model(self, fn):
        np.savetxt(fn, self.weights)


class PartiallyCompletedPlacement(object):
    def __init__(self, p):
        self.p = p
        self.i = 0;

    def is_incomplete(self):
        return self.i < self.p.n()

    def incr_action(self):
        assert self.is_incomplete()
        a = self.p.minor_actions[self.i]
        self.i += 1
        return a

    def last_action(self):
        assert self.i>0
        return self.p.minor_actions[self.i-1]

class FullPlacementActor(object):
    def __init__(self, teacher_iters, QM, e, epsilon):
        self.eps = epsilon
        self.placement = None
        self.penum = PlacementEnumerator(e)
        self.teacher_iters = teacher_iters
        self.teacher = LowestCenterOfGravityAgent(e)
        self.QM = QM

    def act(self, s):
        if s.t < self.teacher_iters:
            return self.teacher.act(s)

        if self.placement is None or not self.placement.is_incomplete():
            choices = self.penum.get_placement_actions(s)
            if random.random()<self.eps:
                p = random.choice(choices)
            else:
                all_actions = map(lambda x: x.minor_actions,choices)
                all_bitmaps = map(lambda x: x.final_state.arena.bitmap[-1].flatten(),choices)
                initial_bitmaps = [s.arena.bitmap[-1].flatten() for i in all_bitmaps]
                #print "{} choices. Bitmaps/Bitmaps/Betas: {}".format(len(choices), list(zip(initial_bitmaps,all_bitmaps,map(lambda x: self.QM.beta(s,x), choices))))
                #raise Exception("TODO")

                p = max(choices, key=lambda x: self.QM.q(s,x))
            self.placement = PartiallyCompletedPlacement(p)
        return self.placement.incr_action()


class FullPlacementEventAggregator(object):
    def __init__(self):
        self.partial_tuple = [None, None, None, None]
        self.ready_items = collections.deque()

    def observe_sars_tuple(self,s,a,r,sprime, pfbm=None):
        if self.partial_tuple[0] is None:
            self.partial_tuple[0] = s
            self.partial_tuple[1] = PlacementAction([],None,None)
            self.partial_tuple[1].add_minor_action(a)
            self.partial_tuple[2] = 0.0
        else:
            self.partial_tuple[1].add_minor_action(a)
            self.partial_tuple[2] += r
            if not np.array_equal(s.arena.bitmap, sprime.arena.bitmap):
                self.partial_tuple[1].set_final_state(sprime)
                self.partial_tuple[1].set_prefinal_bitmap(pfbm)
                self.partial_tuple[3] = sprime
                self.ready_items.append(tuple(self.partial_tuple))
                self.partial_tuple = [None,None,None,None]

    def events_ready(self):
        return len(self.ready_items)>0

    def next_event(self):
        assert self.events_ready()
        return self.ready_items.popleft()


class QLearner(object):
    def __init__(self, QM, config):
       self.QM = QM
       self.items = collections.deque([], config["max_history_len"])
       self.alpha = config["learning_rate"]
       self.delta_norm_clip = config["delta_norm_clip"]
       self.alpha_decay = config["learning_rate_decay"]
       self._lambda = config["eligibility_trace_lambda"]
       self.gamma = config["discount_gamma"]
       self.event_aggregator = json_descr_to_class(config["event_aggregator"])
       self.delta_logger = collections.deque([], maxlen=50)

    def add_history_item(self, event):
        for i in self.items:
            i[1] *= self._lambda * self.gamma
        self.items.append([event,1.0/(self._lambda * self.gamma)]) # Quirky: we start learning from the second to last example

    def observe_sars_tuple(self, s, a, r, sprime, pfbm=None):
        self.event_aggregator.observe_sars_tuple(s,a,r,sprime, pfbm=pfbm)
        while self.event_aggregator.events_ready():
            self.add_history_item(self.event_aggregator.next_event())
            #print("Event: {}".format(self.items[-1]))
            self.learn()
            self.reduce_learning_rate()
        
    def reduce_learning_rate(self):
        self.alpha *= self.alpha_decay
        print "Learning rate: {:.3f}".format(self.alpha)
            
    def learn(self):
        # SARSA with approximation and eligibility trace
        # TODO: this code does not include discounting
        if len(self.items)>1:
            curr = self.items[-2]
            _next = self.items[-1]

            s = curr[0][0]
            a = curr[0][1]
            r = curr[0][2]
            sprime = _next[0][0]
            aprime = _next[0][1]
            
            q = self.QM.q(s,a)
            qprime = self.QM.q(sprime,aprime)

            qact = r + self.gamma * qprime

            delta = qact - q
            self.delta_logger.append(delta)
            logging.info("DELTA: {}".format(delta))
            normdelta = np.linalg.norm(delta)
            if normdelta > self.delta_norm_clip:
                delta = delta * (self.delta_norm_clip / normdelta)

            avgabsdelta = sum([abs(v) for v in self.delta_logger])/len(self.delta_logger)
            avgdelta = sum(self.delta_logger)/len(self.delta_logger)
    
            for i in range(len(self.items)-1):
                s = self.items[i][0][0]
                a = self.items[i][0][1]
                n = self.items[i][1]
                beta = self.QM.beta(s,a)
                self.QM.update_weights(self.alpha * beta * delta * n)
                print " ".join(self.QM.features.names)
                np.set_printoptions(precision=2,linewidth=200)
                print "Qpred: {:9.4f}  Qact: {:9.4f}  Delta: {:9.4f}  Avg50AbsDelta: {:9.4f}  Avg50Delta: {:9.4f}".format(q, qact, delta, avgabsdelta, avgdelta)
                #np.savetxt(sys.stdout, np.hstack((beta, self.QM.weights)).reshape((6,self.QM.weights.size/3)).T, fmt="%8.3f", delimiter=" ")
                #np.savetxt(sys.stdout, np.hstack((beta, self.QM.weights)).reshape((6,self.QM.weights.size/3)).T, fmt="%8.3f", delimiter=" ")


class QLearningAgent(object):
    def __init__(self,e,config):
        self.QM = QModel(config["model"])
        self.QL = QLearner(self.QM, config["learner"])
        self.QA = json_descr_to_class(config["actor"], QM=self.QM, e=e)


    def act(self,s):
        return self.QA.act(s)

    def observe_sars_tuple(self,s,a,r,sprime,pfbm=None):
        self.QL.observe_sars_tuple(s,a,r,sprime,pfbm=pfbm)
        logging.info("REWARD: {}".format(r))

    def print_state(self):
        self.QM.print_state()

    def save_model(self,fn):
        self.QM.save_model(fn)
