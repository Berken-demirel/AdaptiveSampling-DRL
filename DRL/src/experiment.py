import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from Replay_Buffer import Replay_Buffer
from Agent import Agent
from Config import Config
from Environment import Environment
from EpsilonGreedyStrategy import Epsilon_Greedy_Exploration
import time
import csv
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

import copy

config_file_path = "configs/DQL.yaml"

config = Config(config_file_path)

environment = Environment(config, 'eval')

config.num_to_action = {0: '8', 1: '4', 2: '2', 3: '1'}

print(f'name: {config.name}')
print(f'device: {config.device}')
print(f'num_trials: {config.num_trials}')
print(f'replay_batch_size: {config.replay_batch_size}')
print(f'epsilon_decay_rate: {config.epsilon_decay_rate}')
print(f'max_epsilon: {config.max_epsilon}')
print(f'min_epsilon: {config.min_epsilon}')
print(f'learning_rate_min: {config.learning_rate_min}')
print(f'learning_rate_max: {config.learning_rate_max}')
print(f'gamma: {config.gamma}')
print(f'target_network_update_rate: {config.target_network_hard_update_rate}')
print(f'tau: {config.tau}')
print(f'replay_memory_size: {config.replay_memory_size}')

exploration_strategy = Epsilon_Greedy_Exploration(config)

agent = Agent(config, exploration_strategy)

if config.use_PER:
    config_string = f'{config.name}_PER_lr{config.learning_rate_max}_bs{config.replay_batch_size}_trial{config.num_trials}'
else:
    config_string = f'{config.name}_lr{config.learning_rate_max}_bs{config.replay_batch_size}_trial{config.num_trials}'

model_save_path = config.policy_model_save_path + config_string.replace('.', '_') + '.pkl'
model_name = model_save_path.split('/')[-1].split('.')[0]
#model_save_path = config.policy_model_save_path + config.comm_tech + "/" + "Best.pkl"

if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    agent = Agent(config, exploration_strategy)
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    print(f'model path {model_save_path}')
    print(f'DRL model {model_name} loaded')
else:
    print(f'DRL model path {model_save_path} not found')
    exit(-1)

agent.policy_net.to(config.device)
agent.policy_net.eval()

policy_actions = []

current_state = environment.get_state(False)
prev_action = 3

start_time = time.time()
for i in range(config.num_steps_eval):
    with torch.no_grad():
        best_action = agent.select_action(current_state, prev_action)
        reward, current_state = environment.step(best_action)
    policy_actions.append(best_action)
    prev_action = best_action

end_time = time.time()
elapsed_time = end_time - start_time
total_time += elapsed_time

print(f'total_time: {total_time}, average_episode_time: {total_time / episodes_to_run}')

with open('policy_actions.npy', 'wb') as f:
    np.save(f, policy_actions)
