from collections import namedtuple, deque
import random
import torch
import numpy as np
from numpy.random import choice

class SumTreeNode():
    def __init__(self, left, right, is_leaf: bool = False, idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf

def create_tree(input: list):
    nodes = [SumTreeNode.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [SumTreeNode(*pair) for pair in zip(inodes, inodes)]
    return nodes[0], leaf_nodes

def retrieve(value: float, node: SumTreeNode):
    if node.is_leaf:
        return node
    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)

def update(node: SumTreeNode, new_value: float):
    change = new_value - node.value
    node.value = new_value
    propagate_changes(change, node.parent)

def propagate_changes(change: float, node: SumTreeNode):
    node.value += change
    if node.parent is not None:
        propagate_changes(change, node.parent)

class Prioritized_Replay_Buffer():
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, config):
        self.config = config

        self.experiences_per_sampling = self.config.replay_batch_size

        self.memory = deque(maxlen=self.config.replay_memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        #self.memory = [namedtuple("Experience", field_names=["state", "action", "reward", "next_state"]) for i in range(self.config.replay_memory_size)]

        self.base_node, self.leaf_nodes = create_tree([0 for i in range(self.config.replay_memory_size)])
        self.beta = self.config.min_beta
        self.alpha = 0.6
        self.min_priority = 0.01
        self.current_idx = 0

        # self.data = namedtuple("Data", field_names=["priority", "probability", "weight", "index"])
        # indices = []
        # datas = []
        # for i in range(self.config.replay_memory_size):
        #     indices.append(i)
        #     d = self.data(0, 0, 0, i)
        #     datas.append(d)
        # self.memory_dict = {key: self.experience for key in indices}
        # self.memory_data = {key: data for key, data in zip(indices, datas)}

    def update(self, idx, priority):
        update(self.leaf_nodes[idx], self.adjust_priority(priority))

    def adjust_priority(self, prioirty):
        return np.power(prioirty + self.min_priority, self.alpha)

    def add_experience(self, state, action, reward, next_state, priority):
        experience = self.experience(state, action, reward, next_state)
        self.memory.append(experience)
        self.update(self.current_idx, priority)
        self.current_idx += 1
        if self.current_idx >= self.config.replay_memory_size:
            self.current_idx = 0

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences, sampled_idxs, is_weights = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            current_states, actions, rewards, next_states = self.separate_out_data_types(experiences)
            return current_states, actions, rewards, next_states, sampled_idxs, is_weights
        else:
            return experiences, sampled_idxs, is_weights
            
    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        current_states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.config.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.config.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.config.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.config.device)
        return current_states, actions, rewards, next_states
    
    def pick_experiences(self, num_experiences=None):
        sampled_idxs = []
        is_weights = [] #importance sampling weights
        sample_no = 0

        sampled_batch = []
        while sample_no < num_experiences:
            sample_val = np.random.uniform(0, self.base_node.value)
            samp_node = retrieve(sample_val, self.base_node)
            if samp_node.idx < len(self.memory) - 1:
                sampled_idxs.append(samp_node.idx)
                p = samp_node.value / self.base_node.value
                is_weights.append(len(self.memory) * p)
                sample_no += 1
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, -self.beta)
        is_weights = is_weights / np.max(is_weights)
        for idx in sampled_idxs:
            sampled_batch.append(self.memory[idx])

        self.beta = self.config.min_beta + (self.config.max_beta - self.config.min_beta) * min((self.current_idx * self.config.epsilon_decay_rate) / (self.config.num_episodes_train * self.config.num_trials), self.config.max_beta)
        return sampled_batch, sampled_idxs, is_weights

    def __len__(self):
        return len(self.memory)
