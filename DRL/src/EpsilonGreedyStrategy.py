import random
import torch
import numpy as np

class Epsilon_Greedy_Exploration():
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, config):
        self.config = config

    def decay(self, current_episode, num_episodes_to_run):
        #print(self.config.num_episodes_to_run)
        """Gets the probability that we just pick a random action.
        This probability decays the more episodes we have seen"""
        # return self.config.min_epsilon + \
        #        (self.config.max_epsilon - self.config.min_epsilon) * np.exp(-self.config.epsilon_decay_rate * current_episode)
        return self.config.min_epsilon + (self.config.max_epsilon - self.config.min_epsilon)\
               * max((num_episodes_to_run - current_episode * self.config.epsilon_decay_rate) / num_episodes_to_run, 0)
        #return epsilon / (1.0 + (current_episode / self.decay_rate))