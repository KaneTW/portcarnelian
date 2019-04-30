import numpy as np
import random
import math
from tensorforce.environments import Environment

from tensorforce.agents import *
from tensorforce.execution import Runner

class BandedPrinceAction(object):
    def is_available(self, state):
        return state.get_level() == 12 and state.striped > 0
    
    def perform_action(self, state):
        state.progress_cp = 1

        striped = state.striped
        state.striped = 0

        small = round(striped / 30.0) + 2
        big = round(striped / 70.0)

        return 28*(small*2.5 + big*12.5 + 15.625)/29

class EquineAction(object):
    def is_available(self, state):
        return state.get_level() == 12 and state.horseheads > 0
    
    def perform_action(self, state):
        state.progress_cp = 1

        horseheads = state.horseheads
        state.horseheads = 0

        small = round(horseheads / 30.0) + 2
        big = round(horseheads / 70.0)

        return (small*2.5 + big*12.5)


class HonoredAction(object):
    def is_available(self, state):
        return state.get_level() == 12 and state.horseheads > 0 and state.striped > 0
    
    def perform_action(self, state):
        state.progress_cp = 1

        state.horseheads = 0
        state.striped = 0

        return 25

class RandomAction(object):
    def __init__(self, min_level, max_level, min_airs, max_airs, magnitude):
        self.min_level = min_level
        self.max_level = max_level
        self.min_airs = min_airs
        self.max_airs = max_airs
        self.magnitude = magnitude
    
    def is_available(self, state):
        return state.get_level() >= self.min_level and state.get_level() <= self.max_level \
            and state.airs >= self.min_airs and state.airs < self.max_airs

    def perform_action(self, state):
        state.progress_cp += 3
        if random.random() > 0.5:
            state.horseheads += self.magnitude
        else:
            state.striped += self.magnitude
        return 0 - 1.4



class PortCarnelianAction(object):
    def __init__(self, min_level, max_level, min_airs, max_airs, legitimacy_change=0, striped_change=0, horsehead_change=0, echo_cost=0, striped_req=0, horsehead_req=0):
        self.min_level = min_level
        self.max_level = max_level
        self.min_airs = min_airs
        self.max_airs = max_airs
        self.legitimacy_change = legitimacy_change
        self.striped_change = striped_change
        self.horsehead_change = horsehead_change
        self.striped_req = striped_req
        self.horsehead_req = horsehead_req
        self.echo_cost = echo_cost
    
    def is_available(self, state):
        return state.get_level() >= self.min_level and state.get_level() <= self.max_level  \
            and state.airs >= self.min_airs and state.airs <= self.max_airs \
            and state.striped >= self.striped_req and state.horseheads >= self.horsehead_req \
            and state.legitimacy + self.legitimacy_change > 0
    
    def perform_action(self, state):
        state.progress_cp += 3
        state.legitimacy += self.legitimacy_change
        state.striped += self.striped_change
        state.horseheads += self.horsehead_change
        return -self.echo_cost-1.4

ACTIONS = [
      PortCarnelianAction(1, 11, 0, 100, striped_change=4)
    , PortCarnelianAction(1, 11, 0, 100, horsehead_change=4)
    , PortCarnelianAction(1, 10, 1, 10, striped_change=15, legitimacy_change=-10)
    , PortCarnelianAction(1, 10, 1, 10, striped_change=-10, legitimacy_change=5)
    , PortCarnelianAction(1, 10, 91, 100, horsehead_change=15, legitimacy_change=-10)
    , PortCarnelianAction(1, 10, 91, 100, horsehead_change=-10, legitimacy_change=5)
    , PortCarnelianAction(1, 6, 1, 40, striped_change=-20, horsehead_change=25, striped_req=20)
    , PortCarnelianAction(1, 6, 11, 50, striped_change=15, legitimacy_change=-10)
    , PortCarnelianAction(1, 6, 41, 50, legitimacy_change=10)
    , RandomAction(1, 6, 51, 70, 10)
    , PortCarnelianAction(1, 6, 51, 90, horsehead_change=15, legitimacy_change=-10)
    , PortCarnelianAction(1, 6, 71, 100, striped_change=25, horsehead_change=-20, horsehead_req=20)    
    , PortCarnelianAction(7, 10, 1, 30, horsehead_change=15, legitimacy_change=-10)
    # skipping a dangerous source
    , RandomAction(7, 10, 31, 40, 15)
    , PortCarnelianAction(7, 10, 31, 60, horsehead_change=20, legitimacy_change=-15, horsehead_req=10)
    , PortCarnelianAction(7, 10, 41, 70, striped_change=25, legitimacy_change=-20, striped_req=25)
    , PortCarnelianAction(7, 10, 61, 70, legitimacy_change=10)
    , PortCarnelianAction(7, 10, 71, 90, horsehead_change=25, legitimacy_change=-20, horsehead_req=25)
    , PortCarnelianAction(7, 10, 71, 100, striped_change=25, horsehead_change=-20, horsehead_req=20)
    , PortCarnelianAction(11, 11, 1, 20, striped_change=12, echo_cost=2.5)
    , PortCarnelianAction(11, 11, 21, 40, horsehead_change=12, echo_cost=2.5)
    , PortCarnelianAction(11, 11, 41, 50, legitimacy_change=10)
    , PortCarnelianAction(11, 11, 0, 100, striped_change=-30, horsehead_change=38)
    , PortCarnelianAction(11, 11, 0, 100, striped_change=38, horsehead_change=-30)
    , HonoredAction()
    , BandedPrinceAction()
    , EquineAction()
]

class PortCarnelianState(object):
    def __init__(self, progress_cp=1, legitimacy=100, striped=0, horseheads=0, airs=random.randint(1,100)):
        self.progress_cp = progress_cp
        self.legitimacy = legitimacy
        self.striped = striped
        self.horseheads = horseheads
        self.airs = airs
        self.actions = 0
    
    def get_level(self):
        prog = self.progress_cp
        for lv in range(0,13):
            prog = prog - (lv + 1)
            if prog <= 0:
                return lv + 1

    def normalize(self):
        self.legitimacy = max(self.legitimacy, 0)
        self.legitimacy = min(self.legitimacy, 100)
        self.striped = max(self.striped, 0)
        self.horseheads = max(self.horseheads, 0)
    
    def __str__(self):
        return "Level: {}, Legitimacy: {}, Striped: {}. Horseheads: {}, Actions: {}".format(self.get_level(), self.legitimacy, self.striped, self.horseheads, self.actions)


class PortCarnelian(Environment):
    def __str__(self):
        return str(self._state)

    def reset(self):
        self.__init__()
        return self.out_state()

    def out_state(self):
        return [self._state.get_level(), self._state.legitimacy, self._state.striped, self._state.horseheads, self._state.airs]

    def execute(self, action):
        reward = 0

        
        terminal = self._state.legitimacy <= 0
        if terminal:
            return self.out_state(), terminal, reward
        
        action_available = self.is_action_available(action)
        if not action_available:
            return self.out_state(), terminal, reward

        reward = self.do_action(action)

        return self.out_state(), terminal, reward

    @property
    def states(self):
        return dict(shape=5, type='float')
    
    @property
    def actions(self):
        return dict(num_actions=len(ACTIONS), type='int')

    def __init__(self, state=None):
        if state is None:
            self._state = PortCarnelianState()
        else:
            self._state = state


    def is_action_available(self, action):
        return ACTIONS[action].is_available(self._state)


    def do_action(self, action):
        reward = ACTIONS[action].perform_action(self._state)
        self._state.airs = random.randint(1, 100)

        self._state.normalize()
        self._state.actions += 1

        return reward





environment = PortCarnelian()

def quantize(n):
    return round(n / 15)
    
def double_probabilize(a):
    a = np.exp(a[0]/temp) + np.exp(a[1]/temp)
    return a/sum(a)


def probabilize(a):
    a = np.exp(a/temp)
    return a/sum(a)

alpha = 0.8
gamma = 0.9
temp = 10

q_table = np.load('q_table.npz')['arr_0']
n = 0
ep = 0
cum_rew = 0
cum_act = 0
cum_100k_rew = 0
cum_100k_act = 0
cum_100k_max = 0
cum_100k_min = 0
while True:
    environment.reset()
    ep_rew = 0
    ep += 1
    while True:
        state = environment.out_state()

        pos_actions = q_table[state[0]-1, state[1]//5, \
        quantize(state[2]), quantize(state[3]), math.ceil(state[4]/10)]


        probs = double_probabilize(pos_actions)

        while True:
          action = np.random.choice(len(ACTIONS), p=probs)
          if environment.is_action_available(action):
              break
          else:
              pos_actions[0, action] = -5
              pos_actions[1, action] = -5
              probs[action] = 0
              probs = probabilize(probs)


        out_state, terminal, reward = environment.execute(action)

        if reward > 0:
            cum_rew += reward
            cum_100k_rew += reward
            ep_rew += reward
            cum_100k_max = max(reward, cum_100k_max)
            cum_100k_min = min(reward, cum_100k_min)
        cum_act += 1
        cum_100k_act += 1

        state = environment.out_state()
        pos_actions_new = q_table[state[0]-1, state[1]//5, \
        quantize(state[2]), quantize(state[3]), math.ceil(state[4]/10)]
    

        if terminal:
            reward = -2

        dqchoice = random.randrange(2)
        other = 1 - dqchoice
        max_q = np.argmax(pos_actions_new[dqchoice])

        pos_actions[dqchoice, action] = pos_actions[dqchoice, action] + alpha * ( reward + gamma * pos_actions_new[other, max_q] - pos_actions[dqchoice, action])

        if terminal:
            break

        n += 1

        if n % 1000 == 0:
            print("Cum EPA {}, Actions {}, Cum 100k EPA {}, Actions {}".format(cum_rew/cum_act, cum_act, cum_100k_rew/cum_100k_act, cum_100k_act))
            print("100k: EPA MAX {}, EPA MIN {}".format(cum_100k_max, cum_100k_min))
            print("q_table max: {}", np.max(q_table))

        if n % 50000 == 0:
            print(environment)
            print(pos_actions[0])
            print(pos_actions[1])
            
        if n % 100000 == 0:
            cum_100k_act = 0
            cum_100k_rew = 0
            cum_100k_max = 0
            cum_100k_min = 0
        
    if ep % 100 == 0:
        print("Ep {} F, EPA {}, Env {}".format(ep, ep_rew/environment._state.actions, environment))
        

    