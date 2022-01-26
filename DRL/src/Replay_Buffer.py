from collections import namedtuple, deque
import random
import torch
import numpy as np

class Replay_Buffer():
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, config):
        self.config = config
        self.memory = deque(maxlen=self.config.replay_memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "prev_action", "action", "reward", "next_state", "done"])

    def add_experience(self, state, prev_action, action, reward, next_state, done):
        experience = self.experience(state, prev_action, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            current_states, prev_actions, actions, rewards, next_states, done = self.separate_out_data_types(experiences)
            return current_states, prev_actions, actions, rewards, next_states, done
        else:
            return experiences
            
    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        current_states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.config.device)
        prev_actions = torch.from_numpy(np.vstack([e.prev_action for e in experiences if e is not None])).float().to(self.config.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.config.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.config.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.config.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(self.config.device)
        return current_states, prev_actions, actions, rewards, next_states, dones
    
    def pick_experiences(self, num_experiences=None):
        #assert (len(self.memory) >= self.config.replay_batch_size, "batch size > replay buffer size")
        if num_experiences is not None:
            #assert (num_experiences <= len(self.memory), "num_experiences > replay buffer size")
            batch_size = num_experiences
        else: batch_size = self.config.replay_batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
