# import gymnasium as gym
import json
import math
import os
import random
import time
from collections import deque, namedtuple
from itertools import count

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms


def time_trajectory_list(parent_path, file_extension='.json'):
    """ Return a list of '.json' file paths sorted by creation time. """
    npy_files = []
    # Walk through all directories and files in the parent path
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                creation_time = os.path.getctime(file_path)
                npy_files.append((file_path, creation_time))
    # Sort files based on creation time
    npy_files_sorted = sorted(npy_files, key=lambda x: x[1])
    
    # Return only the file paths, sorted by creation time
    return [file[0] for file in npy_files_sorted]

class ReplayMemory(object):
    def __init__(self, capacity = 10000):
        self.memory = deque([], maxlen=capacity)
        
    
    def __len__(self):
        return len(self.memory)

    def save(self,memory_path,trajectory_name,trajectory):
        #save the trajectory as npy in the memory_path
        # create the folder if it does not exist
        if not os.path.exists(memory_path ):
            os.makedirs(memory_path )
        json.dump(trajectory, open(memory_path + '/' + trajectory_name, 'w'), indent=4)


    def sample(self, trajectory_main_path,batch_size):
        trajectorys = []
        trajectory_list = time_trajectory_list(trajectory_main_path)
        if len(trajectory_list) <= batch_size:
            return self.memory

        new_data_ratio = 0.3  # 30% from the newest data
        # old_data_ratio = 1 - new_data_ratio
        new_data_count = int(batch_size * new_data_ratio)
        old_data_count = batch_size - new_data_count  # Ensures total is always batch_size

        # Split memory into 'new' and 'old' segments
        new_data_start = int(len(trajectory_list) * new_data_ratio)  # Starting index for new data
        new_data = trajectory_list[:new_data_start]  # Newest data
        old_data = trajectory_list[new_data_start:]  # Oldest data

        # Sample from both segments
        new_data_samples = random.sample(new_data, new_data_count)
        old_data_samples = random.sample(old_data, old_data_count) if len(old_data) >= old_data_count else old_data

        samples_json_list = new_data_samples + old_data_samples  # Combine samples from new and old data
        # load all the json file
        for json_file in samples_json_list:
            with open(json_file, 'r') as f:
                trajectory = json.load(f)
                trajectorys.append(trajectory)
        return trajectorys


class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Linear(n_observations, 128)
        
        # Separate streams for value and advantage functions
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Only one output for the value stream
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)  # Output one advantage for each action
        )

    def forward(self, x):
        features = F.relu(self.feature_layer(x))
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine the value and advantages to get the final Q-values
        # The advantage values are normalized by subtracting the mean advantage so that the Q values are based around the state value
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
def resnet18_fixed(num_classes, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    # Using a pre-trained ResNet model adapted for grayscale images
    model = models.resnet18(pretrained=True)
    # Modify the first convolutional layer
    original_first_layer = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Initialize the new conv1 layer by averaging the weights of the original RGB channels
    with torch.no_grad():
        model.conv1.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
    # Adjust the fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    return model

def Resnet18_Dueling_DQN(num_classes, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    # Using a pre-trained ResNet model adapted for grayscale images
    model = models.resnet18(pretrained=True)
    # Modify the first convolutional layer to accept grayscale input
    original_first_layer = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)

    num_ftrs = model.fc.in_features
    # EasyMemory()e the fully connected layer with a dueling architecture
    # State Value stream
    model.value_stream = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, 1)  # Scalar output representing V(s)
    )
    # Advantage stream
    model.advantage_stream = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)  # Outputs A(s, a) for each action
    )

    # Override the forward method to implement dueling architecture
    def forward(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)

        value = model.value_stream(x)
        advantages = model.advantage_stream(x)

        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    # EasyMemory()e the default forward method with the dueling one
    model.forward = forward
    model = model.to(device)
    return model

class tip_shaper_DQN_agent():
    def __init__(self, n_actions, device):
        # BATCH_SIZE is the number of transitions sampled from the EasyMemory() buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer        
        self.BATCH_SIZE = 8  #128 64 32
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.001
        self.EPS_DECAY = 100
        self.TAU = 0.005
        self.LR = 1e-4
        self.model_path = './DQN/pre-train_DQN.pth'
        self.memory_path = './DQN/memory'
        self.memory_path_latest = './DQN/memory/2024-05-01 17-18-31'

        self.policy_net = Resnet18_Dueling_DQN(n_actions, device)
        self.target_net = Resnet18_Dueling_DQN(n_actions, device)
        self.policy_net.load_state_dict(torch.load(self.model_path))
        self.target_net.load_state_dict(self.policy_net.state_dict())                     # copy the weights of the policy network to the target network
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)   # AdamW optimizer 
        self.memory = ReplayMemory()
        self.steps_done = 0
        self.n_actions = n_actions
        self.device = device

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                transform = transforms.Compose([
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
                ])
                state = transform(Image.fromarray(state).convert('L')).unsqueeze(0).to(self.device)
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        # if len(self.memory) < self.BATCH_SIZE:
        #     return
        transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

        state_batch_list = []
        action_batch_list = []
        reward_batch_list = []
        next_state_batch_list = []
        transitions = self.memory.sample(self.memory_path_latest,self.BATCH_SIZE)
        if len(transitions) < self.BATCH_SIZE:
            return
        for trajectory in transitions:
            state_batch = transform(Image.fromarray(np.array(trajectory['state'])).convert('L')).unsqueeze(0).to(self.device)
            state_batch_list.append(state_batch)
            action_batch_list.append(torch.tensor(trajectory['action'], dtype=torch.long, device=self.device).unsqueeze(0))
            reward_batch_list.append(torch.tensor(trajectory['reward'], dtype=torch.float32, device=self.device).unsqueeze(0))
            next_state_batch = transform(Image.fromarray(np.array(trajectory['next_state'])).convert('L')).unsqueeze(0).to(self.device)
            next_state_batch_list.append(next_state_batch)
        
        # batch = Transition(*zip(*transitions))

        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(state_batch_list)
        action_batch = torch.cat(action_batch_list).unsqueeze(1)
        reward_batch = torch.cat(reward_batch_list)
        next_states_batch = torch.cat(next_state_batch_list)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken by current policy_net
        policy_net_result = self.policy_net(state_batch)
        state_action_values = policy_net_result.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using Double DQN approach
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # First, select the best action for next states using the current policy network
            best_actions = self.policy_net(next_states_batch).max(1)[1].unsqueeze(1)
            # Second, evaluate the selected actions using the target network
            next_state_values = self.target_net(next_states_batch).gather(1, best_actions).squeeze()  # Double DQN

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_model(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.policy_net.state_dict(), model_path)

    def load_model(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # 1 of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = self.TAU * policy_net_state_dict[key] + (1 - self.TAU) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

if __name__ == '__main__':
    DQN_agent = tip_shaper_DQN_agent(6, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    DQN_agent.load_model('./DQN/pre-train_DQN.pth')
    DQN_agent.optimize_model()
    DQN_agent.update_target_net()
    DQN_agent.save_model('./DQN/train_results/DQN.pth')

    pass