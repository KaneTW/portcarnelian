import numpy as np
import random
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

        return small*2.5 + big*12.5 + 15.625

class EquineAction(object):
    def is_available(self, state):
        return state.get_level() == 12 and state.horseheads > 0
    
    def perform_action(self, state):
        state.progress_cp = 1

        horseheads = state.horseheads
        state.horseheads = 0

        small = round(horseheads / 30.0) + 2
        big = round(horseheads / 70.0)

        return small*2.5 + big*12.5 


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
        return 0



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
            and state.striped >= self.striped_req and state.horseheads >= self.horsehead_req
    
    def perform_action(self, state):
        state.progress_cp += 3
        state.legitimacy += self.legitimacy_change
        state.striped += self.striped_change
        state.horseheads += self.horsehead_change
        return -self.echo_cost


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
        return np.array([self._state.progress_cp, self._state.legitimacy, self._state.striped, self._state.horseheads, self._state.airs])

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








network_spec = [
      dict(type='dense', size=32)
    , dict(type='internal_lstm', size=32)
    , dict(type='dense', size=32)
]


environment = PortCarnelian()

agent = PPOAgent(
    states = environment.states,
    actions = environment.actions,
    network = network_spec,

)

#agent.restore_model(directory='saved')

runner = Runner(agent=agent, environment=environment)

def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    print(runner.environment)


    if r.episode % 100 == 0:
        r.agent.save_model(directory="./saved/")
    return True


runner.run(episodes=30000, max_episode_timesteps=280, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)