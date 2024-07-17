# gym Version: 0.26.2
import random

import gym
import numpy as np
import torch
import torch.nn as nn
from agent import Agent

env = gym.make('CartPole-v1', render_mode='human')
s = env.reset()[0]

EPSILON_DECAY = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FRQUENCY = 10

n_episodes = 5000
n_time_steps = 1000

n_state = len(s)
n_action = env.action_space.n

agent = Agent(n_input= n_state , n_output=n_action)

REWARD_BUFFER = np.empty(shape = n_episodes)

for episode_i in range(n_episodes): 
    episode_reward = 0                                                                                          # Reset the reward
    for step_i in range(n_time_steps):
        epsilon = np.interp( episode_i * n_time_steps + step_i,[0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        random_sample = random.random()

        if random_sample <= epsilon:
            a = env.action_space.sample()                                                                  # Explore , choose a random action
        else:
            a = agent.online_net.act(s)
        
        s_, r, done, truncated, info = env.step(a)
        agent.memo.add_memo(s, a, r, done, s_[0])                                                                  # Store the memory
        observation = s_[0]                                                                                        # Update the observation
        episode_reward += r                                                                                     # Update the reward

        if done:
            s = env.reset()[0]                                                                                     # Reset the environment
            REWARD_BUFFER[episode_i] = episode_reward                                                        # Store the reward
            break

        if np.mean(REWARD_BUFFER[:episode_i]) > 100:
            while True:
                a = agent.online_net.act(s)
                s, r, done, truncated, info = env.step(a)
                env.render()

            if done:
                env.reset()
        
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()

        # Compute the target
        target_q_values = agent.target_net(batch_s_)
        max_target_q_values = target_q_values.max(axis=1, keepdims=True)[0]                                     # Compute the max target Q value
        targets = batch_r + agent.gamma * (1 - batch_done) * max_target_q_values    

        # Compute q_values
        q_values = agent.online_net(batch_s)
        a_q_values = torch.gather(input= q_values, dim = 1,index= batch_a)                                      # Compute the Q value of the action taken

        # Compute the loss
        # print("targets", targets)
        # print("a_q_values", a_q_values)

        loss = nn.functional.smooth_l1_loss(targets, a_q_values)                                                 # Compute the loss

        # Gradient Descent
        agent.optimizer.zero_grad()                                                                             # Zero the gradient
        loss.backward()                                                                                        # Backward pass
        agent.optimizer.step()                                                                                 # Update the weights

    if episode_i % TARGET_UPDATE_FRQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())                                         # Update the target network
        # show the training process
        print(f'Episode: {episode_i}, Reward: {episode_reward}')
        print("AVG Reward: {}".format(np.mean(REWARD_BUFFER)))
        







# env = gym.make('CartPole-v1', render_mode='human')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, r, done, truncated, info = env.step(action)
#         if truncated:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
# env.close()