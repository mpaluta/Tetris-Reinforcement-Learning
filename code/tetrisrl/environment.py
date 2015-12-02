import random 
import json
import numpy as np

# Size of world
# names of shapes
# where to find shapes
# how many time steps
# reward for each event type



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
        assert self.loc.size==2
        self.shape = shape
        self.oindex = oindex
        self.oshape = self.shape.oshapes[self.oindex]

    def coords(self):
        return self.oshape.coords()+self.loc

    def move(self,offset):
        return LocatedShape(self.loc+offset,self.shape,self.oindex)

    def down(self):
        return self.move(np.array([1,0]))

    def left(self):
        return self.move(np.array([0,-1]))

    def right(self):
        return self.move(np.array([0,1]))

    def clockwise_rotate(self):
        return LocatedShape(self.loc,self.shape,(self.oindex+1)%len(self.shape.oshapes))

    def rmin(self):
        return self.oshape.rmin()+self.loc[0]

    def rmax(self):
        return self.oshape.rmax()+self.loc[0]

    def cmin(self):
        return self.oshape.cmin()+self.loc[1]

    def cmax(self):
        return self.oshape.cmax()+self.loc[1]


class Arena(object):
    """Bitmap containing arena state"""
    bitmap=None
    
    def __init__(self,shape=None,bitmap=None):
        if bitmap is not None:
            self.bitmap = bitmap
        else:
            self.bitmap = np.zeros(shape,dtype=np.int8)

    def located_shape_valid(self,ls):
        if ls.rmin()<0 or ls.rmax()>=self.bitmap.shape[0]:
            return False
        if ls.cmin()<0 or ls.cmax()>=self.bitmap.shape[1]:
            return False
        c=ls.coords()
        rs=c[:,0]
        cs=c[:,1]
        if self.bitmap[rs,cs].any():
            return False
        return True

    def add_shape(self, ls):
        assert self.located_shape_valid(ls)
        c=ls.coords()
        rs=c[:,0]
        cs=c[:,1]
        self.bitmap[rs,cs] = 1

    def copy(self):
        return Arena(bitmap=self.bitmap.copy())


class State(object):
    t=None
    arena=None
    lshape=None

    def __init__(self, t, arena, lshape):
        self.t = t
        self.arena = arena
        self.lshape = lshape

    def copy(self):
        b=self.arena.copy()
        return State(self.t, b, self.lshape)


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

        rmin,rmax,cmin,cmax = self.get_start_position_bounds(os)

        loc = np.array([self.random.randrange(rmin,rmax), self.random.randrange(cmin,cmax)])
        return LocatedShape(loc,s,osi)

    def get_start_position_bounds(self,os):
        cmin = 0 - os.cmin()
        cmax = self.arena_dims[1] - os.cmax()
        r = 0 - os.rmin()
        return (r,r+1,cmin,cmax)

    def all_possibilities(self):
        lshapes = []
        for s in self.shapes:
            for i in range(s.num_orientations()):
                os = s.oshapes[i]
                rmin,rmax,cmin,cmax = self.get_start_position_bounds(os)
                all_rc_pairs = itertools.product(range(rmin,rmax),range(cmin,cmax))
                for rc in all_rc_pairs:
                    lshapes.append(LocatedShape(np.array(rc), s, i))
        return lshapes


class RewardStructure(object):

    def __init__(self,config):
        self._move_or_rotate = config["move_or_rotate"]
        self._invalid_move = config["invalid_move"]
        self._time_step = config["time_step"]
        self._rows_cleared = config["rows_cleared"]
        self._game_over = config["game_over"]

    def move_or_rotate(self):
        return self._move_or_rotate
    def invalid_move(self):
        return self._invalid_move
    def time_step(self):
        return self._time_step
    def rows_cleared(self,n):
        return self._rows_cleared[n-1]
    def game_over(self):
        return self._game_over


# os.path.join(os.path.dirname(__file__), "database.dat")
class Environment(object):
    shapes=None
    shapegen=None
    arena_dims=None
    K=None
    R=None

    def __init__(self,config):
        self.arena_dims = config["arena"]["shape"]
        self.shapes = [Shape(n,specs) for n,specs in config["shapes"].iteritems()]
        self.shapegen = LocatedShapeGenerator(self.shapes, self.arena_dims)
        self.K = config["t_fall"]
        self.R = RewardStructure(config["rewards"])

    def initial_state(self):
        s = State(0, Arena(shape=self.arena_dims), self.shapegen.generate())
        return s
        
    def next_state_and_reward(self, s, a):
        sprime = s.copy()
        ls = s.lshape
        r = 0.0
        if a==Action.Left:
            r += self.R.move_or_rotate()
            ls = ls.left()
            if s.arena.located_shape_valid(ls):
                sprime.lshape = ls
            else:
                r += self.R.invalid_move()

        elif a==Action.Right:
            r += self.R.move_or_rotate()
            ls = ls.right()
            if s.arena.located_shape_valid(ls):
                sprime.lshape = ls
            else:
                r += self.R.invalid_move()

        elif a==Action.ClockwiseRotate:
            r += self.R.move_or_rotate()
            ls = ls.clockwise_rotate()
            if s.arena.located_shape_valid(ls):
                sprime.lshape = ls
            else:
                r += self.R.invalid_move()

        elif a==Action.Down:
            while True:
                ls_next = ls.down()
                if s.arena.located_shape_valid(ls_next):
                    ls = ls_next
                else:
                    break
            sprime.lshape = ls
                
        elif a==Action.NoMove:
            pass

        prefinal_bitmap=None

        if s.t % self.K == 0:
            ls_next = s.lshape.down()
            if s.arena.located_shape_valid(ls_next):
                sprime.lshape = ls_next
            else:
                ls_next = None
                ls = s.lshape
                sprime.arena.add_shape(ls)
                row_indices = np.array(sorted(set(ls.coords()[:,0].tolist())),dtype=np.int8)
                complete = row_indices[sprime.arena.bitmap[row_indices].all(axis=1)]
                num_cleared = complete.size
                #print "state={}  prior_bitmap={}  lshape_coords={}  next_bitmap={}  row_indices={}  num_cleared={}".format(sprime, s.arena.bitmap[-1], ls.coords(), sprime.arena.bitmap[-1],row_indices, num_cleared)
                prefinal_bitmap = sprime.arena.bitmap.copy()
                if num_cleared>0:
                    num_rows = sprime.arena.bitmap.shape[0]
                    num_cols = sprime.arena.bitmap.shape[1]
                    mask = np.ones((num_rows,)).astype(np.bool_)
                    mask[complete]=False
                    new_top = np.zeros((num_cleared,num_cols))
                    collapsed_bottom = sprime.arena.bitmap[mask]
                    new_bitmap = np.vstack((new_top,collapsed_bottom))
                    assert new_bitmap.shape == sprime.arena.bitmap.shape
                    sprime.arena.bitmap = new_bitmap
                    r += self.R.rows_cleared(num_cleared)
                
                sprime.lshape = self.shapegen.generate()
                if not sprime.arena.located_shape_valid(sprime.lshape):
                    sprime.arena.bitmap[:] = 0
                    sprime.lshape = self.shapegen.generate()
                    r += self.R.game_over()
        sprime.t += 1
        r += self.R.time_step()
        
        return (sprime,r,prefinal_bitmap)


all_actions=list(range(5))
class Action(object):
    Left,Right,Down,ClockwiseRotate,NoMove = all_actions
    
