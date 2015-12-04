import environment
from environment import Action
import numpy as np
import itertools



class Backpointer(object):
    def __init__(self, a, ss):
        self.a = a
        self.ss = ss

class SearchState(object):
    def __init__(self, t, s, r, bp, pfbm):
        self.t = t
        self.s = s
        self.r = r
        self.bp = bp
        self.pfbm = pfbm
        self._is_final = (bp is not None) and (not np.array_equal(bp.ss.s.arena.bitmap, s.arena.bitmap))

    def __eq_info__(self):
        return (tuple(self.s.lshape.coords().tolist()), tuple(self.s.arena.bitmap.tolist()), self.s.lshape.oindex, self.s.t, self._is_final)

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
            sprime, r, pfbm,_ = self.e.next_state_and_reward(ss.s,a)
            results.append(SearchState(ss.t+1, sprime, r+ss.r, Backpointer(a, ss), pfbm))
        #print "successors_results = {}".format(results)
        return results

class SimpleSearchState(object):
    def __init__(self, t, s, r, bp, pfbm, stage):
        self.t = t
        self.s = s
        self.r = r
        self.bp = bp
        self.pfbm = pfbm
        self.stage = stage
        self._is_final = (bp is not None) and (not np.array_equal(bp.ss.s.arena.bitmap, s.arena.bitmap))

    def __eq_info__(self):
        return (self.stage, tuple(self.s.lshape.coords().tolist()), tuple(self.s.arena.bitmap.tolist()), self.s.lshape.oindex, self.s.t, self._is_final)

    def __hash__(self):
        return hash(self.__eq_info__())

    def __eq__(self, other):
        return self.__eq_info__() == other.__eq_info__()

    def is_final(self):
        return self._is_final

class SimpleTransition(object):
    def __init__(self, e):
        self.e = e

    def successors(self, ss):
        if ss.is_final():
            return []
        results = []
        if ss.stage==0:
            possible_actions = [(Action.Left,1),(Action.Right,2),(Action.ClockwiseRotate,0),(Action.Down,3)]
        elif ss.stage==1:
            possible_actions = [(Action.Left,1),(Action.ClockwiseRotate,1),(Action.Down,3)]
        elif ss.stage==2:
            possible_actions = [(Action.Right,2),(Action.ClockwiseRotate,2),(Action.Down,3)]
        elif ss.stage==3:
            possible_actions = [(Action.NoMove,3)]

        for a,nextstage in possible_actions:
            sprime, r, pfbm, _ = self.e.next_state_and_reward(ss.s,a)
            results.append(SimpleSearchState(ss.t+1, sprime, r+ss.r, Backpointer(a, ss), pfbm, nextstage))
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

        for k,g in itertools.groupby(intermediate_states, key=lambda x:x.__eq_info__()):
            #print "EqInfo: {}".format(k)
            lst = list(g)
            ssmax=max(lst, key=lambda x:x.r)
            self.states.append(ssmax)

class SearchCache(object):
    pass

class Search(object):
    def __init__(self, caching=False):
        if caching:
            self.cache = SearchCache()

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
        beams[0].populate_with_states([SearchState(0, s, 0, None, None)])
        final_states = []
        for i in range(1,1000):
            if beams[i-1].size()==0:
                #print "Breaking search because empty beam at {}".format(i-1)
                break
            else:
                beams.append(Beam(trans))
                beams[-1].populate(beams[-2])
                final_states.extend([ss for ss in beams[-1].states if ss.is_final()])

        #for i,b in enumerate(beams):
        #    print "Beam #{}".format(i)
        #    print "----------------------------"
        #    for j,ss in enumerate(b.states):
        #        print "{})  state={}  t={}  coords={}  bmp={}   isfinal={}".format(j, ss.s, ss.t, ss.s.lshape.coords(), ss.s.arena.bitmap[-1], ss.is_final())

        final_tuples = []
        # Sort and group final states by having the same bitmap
        keyfunc=lambda x:tuple(x.s.arena.bitmap.flatten().tolist())
        final_states = sorted(final_states, key=keyfunc)
        filtered_states = []
        for k,g in itertools.groupby(final_states, key=keyfunc):
            filtered_states.append(min(list(g), key=lambda x:x.t))

        for fs in filtered_states:
            bt = self.backtrace(fs)
            final_tuples.append((bt,fs.s,fs.pfbm))

        return final_tuples


class SimpleSearch(object):
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
        trans = SimpleTransition(e)
        beams.append(Beam(trans))
        beams[0].populate_with_states([SimpleSearchState(0, s, 0, None, None, 0)])
        final_states = []
        for i in range(1,1000):
            if beams[i-1].size()==0:
                #print "Breaking search because empty beam at {}".format(i-1)
                break
            else:
                beams.append(Beam(trans))
                beams[-1].populate(beams[-2])
                final_states.extend([ss for ss in beams[-1].states if ss.is_final()])

        #for i,b in enumerate(beams):
        #    print "Beam #{}".format(i)
        #    print "----------------------------"
        #    for j,ss in enumerate(b.states):
        #        print "{})  state={}  t={}  coords={}  bmp={}   isfinal={}".format(j, ss.s, ss.t, ss.s.lshape.coords(), ss.s.arena.bitmap[-1], ss.is_final())

        final_tuples = []
        # Sort and group final states by having the same bitmap
        keyfunc=lambda x:tuple(x.s.arena.bitmap.flatten().tolist())
        final_states = sorted(final_states, key=keyfunc)
        filtered_states = []
        for k,g in itertools.groupby(final_states, key=keyfunc):
            filtered_states.append(min(list(g), key=lambda x:x.t))

        for fs in filtered_states:
            bt = self.backtrace(fs)
            final_tuples.append((bt,fs.s,fs.pfbm))

        return final_tuples
            

    
class PlacementAction(object):
    def __init__(self, minor_actions, final_state, pfbm):
        self.minor_actions = minor_actions
        self.final_state = final_state
        self.pfbm = pfbm

    def n(self):
        return len(self.minor_actions)

    def add_minor_action(self, a):
        self.minor_actions.append(a)

    def set_final_state(self, s):
        self.final_state = s

    def set_prefinal_bitmap(self, pfbm):
        self.pfbm = pfbm

    def get_prefinal_bitmap(self):
        return self.pfbm

    def get_final_bitmap(self):
        return self.final_state.arena.bitmap


class FastSearch(object):
    def __init__(self):
        pass

    def search(self, e, s):
        pass
        # TODO: return actionseq,finalstate,pfbm tuples
        #os = 

class PlacementEnumerator(object):
    def __init__(self, e):
        self.e = e
        if e.paths_allowed == "simple":
            self.search = SimpleSearch()
        elif e.paths_allowed == "all":
            self.search = Search()
        else:
            raise Exception("Unexpected value for paths_allowed: {}".format(e.paths_allowed))

    def _get_actionseq_finalstate_pairs(self, s):
        return self.search.search(self.e, s)

    def get_placement_actions(self, s):
        tuples = self._get_actionseq_finalstate_pairs(s)
        return [PlacementAction(action,finalstate,pfbm) for action,finalstate,pfbm in tuples]

