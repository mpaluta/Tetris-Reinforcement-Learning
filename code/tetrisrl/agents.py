import pygame
import random
from environment import Action
import logging

class HumanAgent(object):
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

    def observe_sars_tuple(self,s,a,r,sprime,pfbm=None):
        logging.info("REWARD: {}".format(r))
        pass

    def save_model(self,fn):
        pass


class RandomAgent(object):
    actions=None
    def __init__(self):
        self.actions = [Action.NoMove, Action.Left, Action.Right, Action.ClockwiseRotate]

    def act(self,s,debug_mode=False):
        return (random.choice(self.actions),[])

    def observe_sars_tuple(self,s,a,r,sprime,pfbm=None):
        logging.info("REWARD: {}".format(r))
        pass

    def save_model(self,fn):
        pass

