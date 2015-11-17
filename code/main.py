#!/usr/bin/env python

import pygame
from pygame.locals import *
from tetrisrl.environment import Environment,Action
from tetrisrl.baseline import LowestCenterOfGravityAgent
import random
import sys

class Globals(object):
    SCREEN_DIMS=(640,480)

class Colors(object):
    WHITE=(255,255,255)
    GRAY=(128,128,128)
    GREEN=(0,255,0)
    RED=(255,0,0)
    BLUE=(0,0,255)
    BLACK=(0,0,0)

class Agent(object):
    pass

class HumanAgent(Agent):
    def __init__(self):
        pass

    def act(self,s):
        action = Action.NoMove
        for event in pygame.event.get():
            if event.type == KEYDOWN or event.type==KEYUP:
                if event.key == K_UP:
                    action=Action.ClockwiseRotate
                elif event.key == K_DOWN:
                    action=Action.Down
                elif event.key == K_LEFT:
                    action=Action.Left
                elif event.key == K_RIGHT:
                    action=Action.Right
                elif event.key == K_q:
                    pygame.quit()
                    sys.exit()
            elif event.type == KEYUP:
                pass
        return action


class RandomAgent(object):
    actions=None
    def __init__(self):
        self.actions = [Action.NoMove, Action.Down, Action.Left, Action.Right, Action.ClockwiseRotate]
    def act(self,s):
        return random.choice(self.actions)

class Engine(object):
    environment=None
    surfaces=None
    screen=None
    clock=None
    s=None
    agent=None
    
    def __init__(self, environment,agent):
        self.environment=environment
        self.s = self.environment.initial_state()
        self.agent=agent
        pygame.init()
        self.screen=pygame.display.set_mode(Globals.SCREEN_DIMS,0,32)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Tetris")
        self.draw()

    def detect_quit(self):
        if pygame.event.peek(QUIT):
            pygame.quit()
            sys.exit()

    def loop(self):
        self.draw()
        while True:
            time_passed = self.clock.tick(6)
            self.detect_quit()
            (sprime,r) = self.environment.next_state_and_reward(self.s, self.agent.act(self.s))
            self.s = sprime
            self.draw()
            
    def draw(self):
        self.screen.fill(Colors.BLACK)
        b = self.s.arena.bitmap
        ls = self.s.lshape
        w = 20
        for r in range(b.shape[0]):
            for c in range(b.shape[1]):
                rect = (w+(w*c),w+(w*r),w,w)
                if b[r,c]:
                    pygame.draw.rect(self.screen, Colors.GREEN, rect)
                else:
                    pygame.draw.rect(self.screen, Colors.GRAY, rect)
                    
        for coord in ls.coords():
            r,c = (coord[0],coord[1])
            pygame.draw.rect(self.screen, Colors.BLUE, (w+(w*c),w+(w*r),w,w))

        pygame.display.update()

e = Environment("../configs/config.json")
#a = HumanAgent()
#ra = RandomAgent()
lcoga = LowestCenterOfGravityAgent(e)
engine = Engine(e,lcoga)
engine.loop()

