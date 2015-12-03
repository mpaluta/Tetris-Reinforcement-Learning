from environment import State
import numpy as np

class ObservationSerializer(object):
    def __init__(self):
        pass

    def deserialize_json(self, o):
        # return s,a,r,sprime,pfbm tuple
        s = State.deserialize_json(o["s"])
        sprime = State.deserialize_json(o["sprime"])
        a = o["a"]
        r = o["r"]
        pfbm = None if o["pfbm"] is None else np.asarray(o["pfbm"])
        return (s,a,r,sprime,pfbm)

    def serialize_json(self, s, a, r, sprime, pfbm=None):
        so = s.serialize_json()
        sprimeo = sprime.serialize_json()
        pfbmo = pfbm.tolist() if pfbm is not None else None
        return {"s":so, "a": a, "r":r, "sprime":sprimeo, "pfbm":pfbmo }
