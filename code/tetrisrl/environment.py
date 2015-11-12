import random 

# Size of world
# names of shapes
# where to find shapes
# how many time steps
# reward for each event type


# os.path.join(os.path.dirname(__file__), "database.dat")
class Environment(object):
    pass

class LocatedShape(object):
    loc=None
    shape=None
    oindex=None

    def __init__(self,loc,shape,oindex):
        self.loc = tuple(loc)
        self.shape = shape
        self.oindex = oindex

    pass
    # Sparse representation: x,y,orientation

class State(object):
    t=None
    bitmap=None
    lshape=None
    pass
    # Bitmap of existing blocks
    # LocatedShape
    # timestep value t

class Shape(object):
    name=None
    bitmaps=None
    
    def __init__(self, name, spec_strings):
        self.name = name
        self.bitmaps = [self.make_bitmap(s) for s in spec_strings]

    def make_bitmap(self, s):
        lines = s.strip().split(",")
        assert all(len(lines) == len(i) for i in lines)
        rows = [map(int,list(l)) for l in lines]
        return np.array(rows)
        

class ShapeGenerator(object):
    shapes=None
    random=None

    def __init__(self, shapes):
        self.shapes = shapes
        self.random = random.Random(0)

    def generate(self):
        return self.random.choice(self.shapes)

class Transition(object):
    shapegen=None
    K=None

    def __init__(self,shapegen,K):
        self.shapegen=shapegen
        self.K=K

    def successors(self, s, a):
        if s.
        if a==Action.Left:
            pass
        elif a==Action.Right:
            pass
        elif a==Action.Down:
            pass
        elif a==Action.ClockwiseRotate:
            pass
        elif a==Action.NoMove:
            pass
        pass
    # function that given state and action, provides list of state/reward/probability tuples
    # If timestep % K == 0, move down
    # Attempt to follow action
    # 

class TrajectoryEnumerator(object):
    pass
    # Given a start state, returns tree
    # Each node in tree has:
    #    state
    #    list of children of form: (action, prob, reward, node)
    # how many nodes are there? MAXTIMETOFALL x XSIZE x YSIZE x 4[orientations] ~ 60K

class Action(object):
    Left,Right,Down,ClockwiseRotate,NoMove = list(range(5))
    
