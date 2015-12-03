#!/usr/bin/env python

import pygame
from pygame.locals import *
from tetrisrl.environment import Environment,Action
from tetrisrl.baseline import LowestCenterOfGravityAgent
from tetrisrl.agents import HumanAgent, RandomAgent
from tetrisrl.rl import QLearningAgent
from tetrisrl.serialize import ObservationSerializer
import json
import random
import sys
import logging
import os
import shutil
import time


class Globals(object):
    SCREEN_DIMS=(640,480)

class Colors(object):
    WHITE=(255,255,255)
    GRAY=(128,128,128)
    GREEN=(0,255,0)
    RED=(255,0,0)
    BLUE=(0,0,255)
    BLACK=(0,0,0)

class Engine(object):
    environment=None
    surfaces=None
    screen=None
    clock=None
    s=None
    agent=None
    max_time=None
    
    def __init__(self, environment,agent,config,output_dir):
        self.environment=environment
        self.s = self.environment.initial_state()
        self.total_pos_r = 0.0
        self.total_neg_r = 0.0
        self.agent=agent
        self.fps = config["fps"]
        self.show = config["show"]
        self.max_time = config["max_time"]
        self.output_dir = output_dir
        self.oserializer = ObservationSerializer()

        if config["log_observations"]:
            self.obs_log_file = open("{}/observations.json".format(output_dir),"w")
        else:
            self.obs_log_file = None

        if config["replay_observations"]:
            for i in range(config["replay_count"]):
                for fn in config["replay_observations"]:
                    self.replay_observations(fn)

        if self.show:
            pygame.init()
            self.font = pygame.font.SysFont(None, 28)    
            self.screen=pygame.display.set_mode(Globals.SCREEN_DIMS,0,32)
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Tetris")
            self.draw()

    def replay_observations(self, fn):
        with open(fn) as fin:
            print "Replaying from file: {}...".format(fn)
            for l in fin:
                s,a,r,sprime,pfbm = self.oserializer.deserialize_json(json.loads(l))
                self.agent.observe_sars_tuple(s,a,r,sprime,pfbm=pfbm)

    def detect_quit(self):
        if self.show:
            if pygame.event.peek(QUIT):
                pygame.quit()
                sys.exit()

    def loop(self):
        self.draw()
        t = 0
        start = time.clock()
        while True:
            t += 1
            if t % 1000 == 0:
                self.agent.save_model("{}/model.{:06d}iters".format(self.output_dir, t))

            if t > self.max_time:
                break

            if self.show:
                self.clock.tick(self.fps)
            self.detect_quit()
            a = self.agent.act(self.s)
            sprime,r,pfbm = self.environment.next_state_and_reward(self.s, a)
            if r > 0:
                self.total_pos_r += r
            else:
                self.total_neg_r += r

            if self.obs_log_file:
                self.obs_log_file.write("{}\n".format(json.dumps(self.oserializer.serialize_json(self.s,a,r,sprime,pfbm=pfbm))))

            self.agent.observe_sars_tuple(self.s,a,r,sprime,pfbm=pfbm)
            self.s = sprime
            self.draw()
            duration = time.clock()-start
            print "Runtime={:.2f}s  T={}  Total Reward: {:.2f}  {:.2f}".format(duration, t, self.total_pos_r, self.total_neg_r)
            
    def draw(self):
        if not self.show:
            return

        self.screen.fill(Colors.BLACK)
        b = self.s.arena.bitmap
        ls = self.s.lshape
        w = 20

        text = self.font.render("Total Reward: {:.2f}  {:.2f}".format(self.total_pos_r, self.total_neg_r), True, Colors.WHITE, Colors.BLUE)
        textRect = text.get_rect()
        textRect.centerx = (w * b.shape[1]) + 250
        textRect.centery = self.screen.get_rect().centery
        self.screen.blit(text, textRect)

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




config_file = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

log_file = "{}/log".format(output_dir)
logging.basicConfig(filename=log_file, filemode="w", level=logging.DEBUG)

saved_config_file = "{}/config".format(output_dir)
shutil.copyfile(config_file, saved_config_file)



with open(config_file,"r") as fin:
    config = json.load(fin)

e = Environment(config["environment"])

agent_type = config["agent"]["type"]

if agent_type == "rl":
    agent = QLearningAgent(e,config["agent"])
elif agent_type == "lcog":
    agent = LowestCenterOfGravityAgent(e)
elif agent_type == "human":
    agent = HumanAgent()
    assert config["engine"]["show"]
elif agent_type == "random":
    agent = RandomAgent()
else:
    raise Exception("Unknown agent type: {}".format(config["agent"]["type"]))

engine = Engine(e,agent,config["engine"],output_dir)
engine.loop()

