import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)
      self.ptr = (self.ptr + 1) % self.max_size

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_views,batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [],[], [], [], [], []
    for i in ind:
      view,state, next_state, action, reward, done = self.storage[i]
      batch_views.append(np.array(view,copy=False))
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_views),np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim+1, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.flattened = nn.Flatten()
        self.dense_layer = nn.Linear(36992, action_dim)
        self.max_action = max_action

        self.max_action = max_action

    def forward(self, s,x):
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = self.dense_layer(self.flattened(s))
        s = s.squeeze(0)
        # try:
        try:
            if x.shape[1]!=1:
                pass
        except:
            x = x[None,:]
            s = s[None,:]
        # print(x.shape)
        # print(s.shape)
        x = torch.cat([x, s], 1)
        # except:
        #     print(x.shape)
        #     print(s.shape)
        #     ipdb.set_trace()
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim+1 + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim+1 + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.flattened_1 = nn.Flatten()
        self.dense_layer_1 = nn.Linear(36992, 300)
        self.dense_layer_3 = nn.Linear(300, 1)
        self.linear_1 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.conv4 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.flattened_2 = nn.Flatten()
        self.dense_layer_2 = nn.Linear(36992, 300)
        self.dense_layer_4 = nn.Linear(300, 1)
        self.linear_2 = nn.Linear(300, 1)

    def forward(self, s, x, u):
        # print(x.shape)
        # print(u.shape)
        try:
            if u.shape[1]!=1:
                pass
        except:
            u = u[:,None]
        # import ipdb;
        # ipdb.set_trace()
        s1 = F.relu(self.bn1(self.conv1(s)))
        s1 = F.relu(self.bn2(self.conv2(s1)))
        s1 = F.relu(self.bn3(self.conv3(s1)))
        s1 = self.flattened_1(s1)

        s1 = self.dense_layer_1(s1)
        s1 = self.dense_layer_3(s1)
        xu1 = torch.cat([s1, x, u], 1)
        # s1 = self.linear_1(x1)
        # Forward-Propagation on the second Critic Neural Network
        s2 = F.relu(self.bn1(self.conv1(s)))
        s2 = F.relu(self.bn2(self.conv2(s2)))
        s2 = F.relu(self.bn3(self.conv3(s2)))
        s2 = self.flattened_1(s2)

        s2 = self.dense_layer_2(s2)
        s2 = self.dense_layer_4(s2)
        xu2 = torch.cat([s2, x, u], 1)


        # xu = torch.cat([x, u], 1)
        # print(xu.shape)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu1))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu2))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, s,x, u):
        # xu = torch.cat([x, u], 1)
        # x1 = F.relu(self.layer_1(xu))
        # x1 = F.relu(self.layer_2(x1))
        # x1 = self.layer_3(x1)
        s1 = F.relu(self.bn1(self.conv1(s)))
        s1 = F.relu(self.bn2(self.conv2(s1)))
        s1 = F.relu(self.bn3(self.conv3(s1)))
        s1 = self.flattened_1(s1)

        s1 = self.dense_layer_1(s1)
        s1 = self.dense_layer_3(s1)
        xu1 = torch.cat([s1, x, u], 1)
        x1 = F.relu(self.layer_1(xu1))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1


class TD3(object):

    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=0.01)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=0.01)
        self.max_action = max_action

    def select_action(self,view_state):
        view = view_state[0]
        state = view_state[1]
        state = torch.Tensor(state).to(device)
        view = torch.Tensor(view).to(device)
        try:
            view = view[None, None, :, :]
        except:
            ipdb.set_trace()
        return self.actor(view,state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_views,batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                batch_size)
            # import ipdb;ipdb.set_trace()
            views = torch.Tensor(batch_views).to(device)
            views = views[:, None, :, :]
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            try:
                action = torch.Tensor(batch_actions).to(device)
                # import ipdb;
                # ipdb.set_trace()
            except:
                # import ipdb;ipdb.set_trace()
                batch_actions = [np.float(x) for x in batch_actions]
                action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)


            # Step 5: From the next state s’, the Actor target plays the next action a’

            next_action = self.actor_target(views,next_state)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            noise = noise.unsqueeze(-1)

            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(views,next_state, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)
            # print(target_Q)
            # print(reward)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(views,state, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # print(critic_loss)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(s=views,x=state, u=self.actor(views,state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

