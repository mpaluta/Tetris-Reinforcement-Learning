import environment
import numpy as np
import itertools


class Backpointer(object):
    def __init__(self, a, ss):
        self.a = a
        self.ss = ss

class SearchState(object):
    def __init__(self, s, r, bp):
        self.s = s
        self.r = r
        self.bp = bp
        self._is_final = (bp is not None) and (not np.array_equal(bp.ss.s.arena.bitmap, s.arena.bitmap))

    def __eq_info__(self):
        return (tuple(self.s.lshape.coords().tolist()), self.s.t, self._is_final)

    def __hash__(self):
        return hash(self.__eq_info__())

    def __eq__(self, other):
        return self.__eq_info__() == other.__eq_info__()

    def is_final(self):
        return self._is_final

class Transition(object):
    def __init__(self, e):
        self.e = e

    def successors(self, ss):
        if ss.is_final():
            #print "successors = []"
            return []
        results = []
        for a in environment.all_actions:
            sprime, r = self.e.next_state_and_reward(ss.s,a)
            results.append(SearchState(sprime,r+ss.r, Backpointer(a, ss)))
        #print "successors_results = {}".format(results)
        return results

class Beam(object):
    def __init__(self,trans):
        self.states = []
        self.trans = trans

    def size(self):
        return len(self.states)

    def populate_with_states(self, states):
        self.states = list(states)

    def populate(self, prev_beam):
        intermediate_states = []
        for ss in prev_beam.states:
            intermediate_states.extend(self.trans.successors(ss))
        #print "Intermediate states: {}".format(intermediate_states)
        intermediate_states = list(sorted(intermediate_states, cmp=lambda x,y:(cmp(x.__eq_info__(),y.__eq_info__()))))
        self.states = []

        for k,g in itertools.groupby(intermediate_states, lambda x:x.__eq_info__()):
            lst = list(g)
            ssmax=max(lst, key=lambda x:x.r)
            self.states.append(ssmax)
        #print "States={}".format(self.states)

class Search(object):
    def __init__(self):
        pass

    def backtrace(self, ss):
        reverse_actions = []
        while ss is not None:
            if ss.bp is not None:
                reverse_actions.append(ss.bp.a)
                ss = ss.bp.ss
            else:
                return list(reversed(reverse_actions))

    def search(self, e, s):
        beams = []
        trans = Transition(e)
        beams.append(Beam(trans))
        beams[0].populate_with_states([SearchState(s,0,None)])
        final_states = []
        for i in range(1,1000):
            if beams[i-1].size()==0:
                #print "Breaking search because empty beam at {}".format(i-1)
                break
            else:
                beams.append(Beam(trans))
                beams[-1].populate(beams[-2])
                final_states.extend([ss for ss in beams[-1].states if ss.is_final()])

        final_tuples = []
        for fs in final_states:
            bt = self.backtrace(fs)
            final_tuples.append((fs.s,bt,fs.r))

        return final_tuples
            

    

class PlacementEnumerator(object):
    def __init__(self):
        pass

    def get_successor_bitmaps(self, e, s):
        search = Search()
        tuples = search.search(e,s)
        return tuples

