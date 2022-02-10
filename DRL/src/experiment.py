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

action_to_decimation = {0: '8', 1: '4', 2: '2', 3: '1'}

print(f'name: {config.name}')
print(f'device: {config.device}')
print(f'num_trials: {config.num_trials}')
print(f'num_subjects_train: {config.num_subjects_train}')
print(f'num_subjects_eval: {environment.num_subjects_eval}')
print(f'replay_batch_size: {config.replay_batch_size}')
print(f'epsilon_decay_rate: {config.epsilon_decay_rate}')
print(f'max_epsilon: {config.max_epsilon}')
print(f'min_epsilon: {config.min_epsilon}')
print(f'learning_rate_min: {config.learning_rate_min}')
print(f'learning_rate_max: {config.learning_rate_max}')
print(f'gamma: {config.gamma}')
print(f'hard_update: {config.hard_update}')
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
patient_wise_policy_actions = {}

optimal_actions = []
patient_wise_optimal_actions = []

total_time = 0
start_time = time.time()

# for i in range(environment.num_steps_eval):
for subject in range(environment.num_subjects_eval):
    environment.done = 0
    environment.subject_index = subject
    current_state = environment.get_state(False)
    prev_action = 3
    patient_wise_list = []
    counter = 0
    while environment.done == 0:
        optimal_actions.append(np.argmax(environment.data_eval[subject][2][counter]))
        with torch.no_grad():
            current_state = torch.tensor([current_state], dtype=torch.float, device=config.device)
            prev_action = torch.tensor([[prev_action]], dtype=torch.float, device=config.device)
            action_values = agent.policy_net(current_state, prev_action)
            best_action = torch.argmax(action_values)
            _, next_state = environment.step(best_action)
        policy_actions.append(best_action.item())
        patient_wise_list.append(best_action.item())
        prev_action = best_action
        current_state = next_state
        counter += 1
    patient_wise_policy_actions[subject] = patient_wise_list
end_time = time.time()
elapsed_time = end_time - start_time
total_time += elapsed_time

action_count = len(policy_actions)

print(f'total_time: {total_time}, average_step_time: {total_time / action_count}')

optimal_action_count = len(optimal_actions)
optimal_action_counts = Counter(optimal_actions)
print(f'optimal_action_counts: {optimal_action_counts}')
print(f'optimal_action_ratios: {np.round(np.asarray(list(optimal_action_counts.values()))/optimal_action_count*100, 2)}')

action_counts = Counter(policy_actions)
print(f'action_counts: {action_counts}')
print(f'action_ratios: {np.round(np.asarray(list(action_counts.values()))/action_count*100, 2)}')
print(f'policy_actions length: {action_count}')
print(f'policy_actions: {policy_actions}')
print(f'patient_wise_actions: {patient_wise_policy_actions}')

action_accuracy = 0
for idx in range(optimal_action_count):
    if policy_actions[idx] == optimal_actions[idx]:
        action_accuracy += 1
action_accuracy = round(action_accuracy/optimal_action_count * 100, 2)
print(f'action_accuracy: {action_accuracy}')

with open(f'results/eval/{model_name}_action_trajectory.npy', 'wb') as f:
    np.save(f, policy_actions)

with open(f'results/eval/{model_name}_patient_wise_action_trajectory.npy', 'wb') as f:
    np.save(f, patient_wise_policy_actions)

print("EVAL")