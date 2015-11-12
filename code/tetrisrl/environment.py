import random 

# Size of world
# names of shapes
# where to find shapes
# how many time steps
# reward for each event type


# os.path.join(os.path.dirname(__file__), "database.dat")
class Environment(object):
    pass

class OrientedShape(object):
    bitmap=None
    _coords=None
    _rbounds=None
    _cbounds=None

    def __init__(self, bitmap):
        self.bitmap = bitmap
        self._coords = np.transpose(np.nonzero(bitmap))
        self._rbounds = np.array([self._coords[:,0].min(),self._coords[:,0].max()])
        self._cbounds = np.array([self._coords[:,1].min(),self._coords[:,1].max()])

    def coords(self):
        return self._coords

    def rmin(self):
        return self._rbounds[0]

    def rmax(self):
        return self._rbounds[1]

    def cmin(self):
        return self._cbounds[0]

    def cmax(self):
        return self._cbounds[1]

class Shape(object):
    name=None
    oshapes=None
    
    def __init__(self, name, spec_strings):
        self.name = name
        self.oshapes = [self._make_oshape(s) for s in spec_strings]

    def _make_oshape(self, s):
        lines = s.strip().split(",")
        assert all(len(lines) == len(i) for i in lines)
        rows = [map(int,list(l)) for l in lines]
        return OrientedShape(np.array(rows))

    def num_orientations(self):
        return len(self.oshapes)


class LocatedShape(object):
    loc=None
    shape=None
    oindex=None
    oshape=None

    def __init__(self,loc,shape,oindex):
        self.loc = np.array(loc)
        assert self.loc.size()==2
        self.shape = shape
        self.oindex = oindex
        self.oshape = self.shape.oshapes[self.oindex]

    def coords(self):
        return self.oshape.coords()+loc

    def move(self,offset):
        return LocatedShape(self.loc+offset,self.shape,self.oindex)

    def down(self):
        return self.move(np.array([1,0]))

    def left(self):
        return self.move(np.array([0,-1]))

    def right(self):
        return self.move(np.array([0,1]))

    def rotate(self):
        return LocatedShape(self.loc,self.shape,(self.oindex+1)%len(self.shape.oshapes))

    def rmin(self):
        return self.oshape.rmin()+self.loc[0]

    def rmax(self):
        return self.oshape.rmax()+self.loc[0]

    def cmin(self):
        return self.oshape.cmin()+self.loc[1]

    def cmax(self):
        return self.oshape.cmax()+self.loc[1]


class Bitmap(object):
"""Bitmap containing arena state"""
    bits=None
    
    def __init__(self,nrows,ncols):
        self.bits = np.zeros((nrows,ncols))

    def located_shape_valid(self,ls):
        if ls.rmin()<0 or ls.rmax()>=bits.shape[0]:
            return False
        if ls.cmin()<0 or ls.cmax()>=bits.shape[1]:
            return False
        c=ls.coords()
        rs=c[:,0]
        cs=c[:,1]
        if bits[rs,cs].any():
            return False
        return True

    def add_shape(self, ls):
        assert self.located_shape_valid(ls)
        c=ls.coords()
        rs=c[:,0]
        cs=c[:,1]
        bits[rs,cs] = 1


class State(object):
    t=None
    bitmap=None
    lshape=None
    pass
    # Bitmap of existing blocks
    # LocatedShape
    # timestep value t


class LocatedShapeGenerator(object):
    shapes=None
    random=None
    arena_dims=None

    def __init__(self, shapes, arena_dims):
        self.shapes = shapes
        self.random = random.Random(0)
        self.arena_dims = arena_dims

    def generate(self):
        s = self.random.choice(self.shapes)
        osi = self.random.randrange(s.num_orientations())
        os = s.oshapes[osi]
        
        cmin = 0 - os.cmin()
        cmax = self.arena_dims[1] - 1 - os.cmax()
        r = 0 - os.rmin()

        loc = np.array([r, self.random.randrange(cmin,cmax+1)])
        #TODO

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
    
