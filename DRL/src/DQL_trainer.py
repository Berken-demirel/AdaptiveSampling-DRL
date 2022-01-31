import torch
import torch.nn as nn
import numpy as np
import yaml
import os
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

def huber_loss(loss):
    loss = torch.where(abs(loss) < 1.0, 0.5 * loss ** 2, abs(loss) - 0.5)
    return loss

def mse_loss(loss):
    return loss ** 2

def plot(x, fig_name, y_label):
    #clear_output(True)
    #plt.clf()
    plt.figure(figsize=(20, 5))
    plt.plot(x)
    plt.ylabel(y_label)
    plt.title(fig_name)
    plt.savefig(fig_name)
    #plt.show()

def plot_bar(data1, file_path, y_label):

    plt.clf()

    fig = plt.figure(figsize=(35, 10))
    ax = fig.add_subplot(111)
    width = 0.4 # the width of the bars

    data1_vals = np.arange(len(data1))
    rects1 = ax.bar(data1_vals, data1, width, align='center', label=file_path.split("/")[-1].split(".")[0].split("_")[-3])
    for i, v in enumerate(data1):
        str_v = "{:.3f}".format(v)
        ax.text(data1_vals[i], v, str_v, horizontalalignment='center')

    plt.title(file_path.split("/")[-1].split(".")[0])
    ax.set_xticks(data1_vals)
    ax.legend(loc='best')

    plt.xlabel('Clips')
    ax.set_ylabel(y_label)

    #plt.legend((rects1[0], rects2[0]), ('Men', 'Women'))

    #plt.subplot(212)

    plt.tight_layout()

    plt.savefig(file_path)

config_file_path = "configs/DQL.yaml"

config = Config(config_file_path)

environment = Environment(config, 'train')
environment.case = 'DRL'

exploration_strategy = Epsilon_Greedy_Exploration(config)

# if os.path.exists('model/DDQN.pkl'):
#     checkpoint = torch.load(config.policy_model_save_path)
#     agent = Agent(config, exploration_strategy)
#     agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
#     agent.epsilon = checkpoint['epsilon']
#     # need to figure out how to checkpoint the replay memory with namedtuples
#     #agent.replay_buffer.memory = checkpoint['replay_buffer_memory']
#     print('DDQN Model loaded')
# else:
#     agent = Agent(config, exploration_strategy)

agent = Agent(config, exploration_strategy)

agent.policy_net.to(config.device)
agent.target_net.to(config.device)
agent.target_net.eval()

action_to_decimation = {0: '8', 1: '4', 2: '2', 3: '1'}

eval_steps = 1

total_time = 0

train_results_y = []

results = []

losses = []

avg_reward_list = []

if config.use_PER:
    config_string = f'{config.name}_PER_lr{config.learning_rate_max}_bs{config.replay_batch_size}_trial{config.num_trials}'
else:
    config_string = f'{config.name}_lr{config.learning_rate_max}_bs{config.replay_batch_size}_trial{config.num_trials}'

print(f'name: {config.name}')
print(f'device: {config.device}')
print(f'num_trials: {config.num_trials}')
print(f'num_subjects_train: {config.num_subjects_train}')
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

#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name('cuda:1'))

torch.set_printoptions(profile="full")

optimal_actions = []
eval_actions = []

start_time = time.time()
for trial in range(config.num_trials+1):
    print(f'Trial {trial+1}')
    environment = Environment(config, 'train')
    environment.reset()
    environment.case = 'DRL'
    prev_action = 3
    if trial == config.num_trials:
        agent.epsilon = 0
    for subject in range(config.num_subjects_train):
        environment.done = 0
        environment.subject_index = subject
        current_state = environment.get_state(False)
        per_subject_reward = []
        #print(f'environment.num_data_segments_train: {environment.num_data_segments_train}')
        counter = 0
        while environment.done == 0:
            if trial == 0:
                optimal_actions.append(np.argmax(environment.data_train[subject][2][counter]))
            action = agent.select_action(current_state, prev_action)
            if trial == config.num_trials:
                eval_actions.append(action)
            reward, next_state = environment.step(action)
            per_subject_reward.append(reward)
            # print(f'current_state: {current_state}')
            # print(f'prev_action: {prev_action}')
            # print(f'action: {action}')
            # print(f'reward: {reward}')
            # print(f'next_state: {next_state}')

            if config.use_PER:
                current_state = torch.tensor(current_state, dtype=torch.float, device=config.device).unsqueeze(0)
                prev_action = torch.tensor(prev_action, dtype=torch.float, device=config.device).unsqueeze(0)
                with torch.no_grad():
                    max_actions = agent.policy_net(next_state, actions).detach().argmax(dim=1).unsqueeze(1)
                    next_q_values = agent.target_net(next_state, actions).gather(dim=1, index=max_actions).squeeze(1)
                    target_q_value = reward + (1-environment.done)*(config.gamma * next_q_values)
                    current_q_value = agent.policy_net(current_state, prev_action)[0][action]
                    #print(current_q_value)
                error = abs(current_q_value - target_q_value)
                agent.replay_buffer.add_experience(current_state, prev_action, action, reward, next_state, error)
            else:
                if environment.done == 1:
                    print(f'subject_num: {subject}')
                    print(f'environment.num_data_segments_train: {environment.num_data_segments_train}')
                    agent.replay_buffer.add_experience(current_state, prev_action, action, reward, next_state, 1)
                else:
                    agent.replay_buffer.add_experience(current_state, prev_action, action, reward, next_state, 0)

            # sample random minibatch
            if len(agent.replay_buffer.memory) >= config.batch_delay*config.replay_batch_size:
                if config.use_PER:
                    current_states, actions, rewards, next_states, sampled_idxs, is_weights = agent.replay_buffer.sample(config.replay_batch_size)
                else:
                    current_states, prev_actions, actions, rewards, next_states, dones = agent.replay_buffer.sample(config.replay_batch_size)

                with torch.no_grad():
                    temp = agent.policy_net(next_states, actions)
                    max_actions = agent.policy_net(next_states, actions).detach().argmax(dim=1).unsqueeze(1)
                    #print(f'temp: {temp}')
                    #print(f'max_actions: {max_actions}')
                    next_q_values = agent.target_net(next_states, actions).gather(dim=1, index=max_actions).squeeze(1)
                    target_q_values = rewards.squeeze(1) + np.multiply((1-dones).squeeze(1), (config.gamma * next_q_values))
                current_q_values = agent.policy_net(current_states, prev_actions).gather(dim=1, index=actions).squeeze(1)

                # print(f'target_q_values:{target_q_values}')
                # print(f'current_q_values:{current_q_values}')

                # q_val_estimate = agent.policy_net(features_current, channel_rates_current, queue_states_current)
                # print(f'q_val_estimate: {q_val_estimate}')
                # print(f'actions: {actions}')
                # gathered_qval = q_val_estimate.gather(dim=1, index=actions).squeeze(1)
                # print(f'gathered_qval: {gathered_qval}')
                # #exit(-1)
                #print(f'target_q_values: {target_q_values}')
                #print(f'current_q_values: {current_q_values}')
                # print(target_q_values.shape)
                # print(current_q_values.shape)
                # exit(-1)
                #print(f'current_q_values: {current_q_values}, target_q_values: {target_q_values}')

                if config.use_PER:
                    # loss = nn.functional.huber_loss(current_q_values, target_q_values) * torch.mean(torch.FloatTensor(is_weights))
                    errors = abs(current_q_values - target_q_values)
                    for j in range(len(sampled_idxs)):
                        agent.replay_buffer.update(sampled_idxs[j], errors[j].item())
                    loss = torch.sum(mse_loss(errors) * torch.FloatTensor(is_weights))
                    # loss_1 = huber_loss(errors)
                    # isamp = torch.FloatTensor(is_weights)
                    # loss = torch.mean(loss_1 * isamp)
                else:
                    #print(f'current_q_values: {current_q_values.shape}')
                    #print(f'target_q_values: {target_q_values.shape}')
                    loss_func = nn.MSELoss()
                    loss = loss_func(current_q_values, target_q_values)
                    #loss = target_q_values - current_q_values
                    #print(f'loss: {loss}')
                losses.append(loss.item())

                agent.optimizer.zero_grad()
                loss.backward()
                # print(agent.policy_net.fc5.weight.grad)

                if config.gradient_clip:
                    for param in agent.policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                agent.optimizer.step()
                #agent.scheduler.step()

                # updates every period
                if config.hard_update:
                    if (counter+1) % config.target_network_hard_update_rate == 0:
                        # hard copy model parameters to target model parameters
                        agent.target_net.load_state_dict(agent.policy_net.state_dict())
                else:
                    # soft update every step
                    for target_param, policy_param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
                        #target_param.data.copy_(policy_param)
                        target_param.data.copy_(config.tau * policy_param.data + (1 - config.tau) * target_param.data)
            prev_action = action
            current_state = next_state
            counter += 1
        avg_reward_list.append((sum(per_subject_reward) / environment.num_data_segments_train))

    # epsilon greedy exploration
    if trial < config.num_trials:
        agent.epsilon = agent.epsilon_decay(trial+1, config.num_trials)
        print(f'epsilon: {agent.epsilon}')

end_time = time.time()
elapsed_time = end_time - start_time
total_time += elapsed_time

model_save_path = config.policy_model_save_path
file_name = config_string.replace('.', '_') + '.pkl'

full_path = model_save_path + file_name

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
torch.save({'model_state_dict': agent.policy_net.state_dict()}, full_path)

print(f'total_time: {total_time}, average_trial_time: {total_time/config.num_trials}')

optimal_action_count = len(optimal_actions)
optimal_action_counts = Counter(optimal_actions)
print(f'optimal_action_counts: {optimal_action_counts}')
print(f'optimal_action_ratios: {np.round(np.asarray(list(optimal_action_counts.values()))/optimal_action_count*100, 2)}')

eval_action_count = len(eval_actions)
eval_action_counts = Counter(eval_actions)
print(f'eval_action_counts: {eval_action_counts}')
print(f'eval_action_ratios: {np.round(np.asarray(list(eval_action_counts.values()))/eval_action_count*100, 2)}')

action_accuracy = 0
for idx in range(optimal_action_count):
    if eval_actions[idx] == optimal_actions[idx]:
        action_accuracy += 1
action_accuracy = round(action_accuracy/optimal_action_count * 100, 2)
print(f'action_accuracy: {action_accuracy}')

print("TRAIN")

if not os.path.exists('results/train/'):
    os.makedirs('results/train/')

plot(avg_reward_list, f'results/train/{full_path.split("/")[-1].split(".")[0]}_rewards.png', 'reward')
plot(losses, f'results/train/{full_path.split("/")[-1].split(".")[0]}_losses.png', 'loss')
