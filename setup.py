""" Sets up the parameters and static structures used throughout learning """

import gym
from gym import wrappers

from helper import preprocess_image

#Main function
def setup():
    """ Sets up static structures and parameters """

    parameter = {
        "GAMMA": .99,
        "EPS": 0.1, #1. for production
        "NUM_EPISODES": 10, #max sohuld be 10 000 or 100 000 for complex tasks #maybe have multiple NUM_EPISODES IF TESTING OR STH ELSE
        "NUM_STEPS": 100, #should be open ended if wrapper is used
        "SAVE_EVERY": 2,
        "OLD_TF": False,
        "SAVE_FIGS": False,
        "X11": True
    }

    env = create_environment(False)

    return env, parameter


def create_environment(record=False):
    """ Creating the environment for the gym 
        In: record (save video of episode every few times
        OUt: env (environment)
    """
    env = gym.make('SpaceInvaders-v0')
    if record:
        env = wrappers.Monitor(env, 'SpaceInvaderExperiment')

    return env


def initialize_replay_memory(N, env):
    observation = env.reset()
    out = []
    for i in xrange(4):
        action = env.action_space.sample()
        new_observation, reward, done, _ = env.step(action)
        p_observation = preprocess_image(observation)
        p_new_observation =preprocess_image(new_observation)
        aptuple = (p_observation, action, reward, p_new_observation)
        out.append( aptuple )
        new_observation = observation

    return out, new_observation

