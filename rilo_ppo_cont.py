from collections import namedtuple

import os, time
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from rilo_utils import torch_RMSE

envname = 'Hopper-v2' # TZY: change from v1 to v2
env_1 = gym.make(envname)
# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10
a_type = 'PPO'
torch.manual_seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self, num_state, num_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = self.action_head(x)
        return action


class Critic(nn.Module):
    def __init__(self, num_state):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.4
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self, num_state, num_action):
        super(PPO, self).__init__()

        self.actor_net = Actor(num_state, num_action)
        self.critic_net = Critic(num_state)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('../exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-2)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-2)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        #c = Categorical(action_prob)
        #action = c.sample()
        return action_prob

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1,3)
        old_action_log_prob = torch.tensor([t.action for t in self.buffer], dtype=torch.float)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        # old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updating....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    rew = 0
                    # evaluate the model at current step
                    for i in range(10):
                        terminal = False
                        total_reward = 0
                        state_p = env_1.reset()
                        cnt = 0
                        while not terminal:

                            if a_type == 'PPO':
                                a = self.select_action(state_p)
                            elif a_type == 'random':
                                a = env_1.action_space.sample()

                            state_p, reward_p, terminal, _ = env_1.step(a)
                            total_reward += reward_p
                            cnt += 1
                        rew += total_reward / 10
                        # print('count:{}, total reward from supervisor: {}'
                        #       .format(cnt, np.round(total_reward, 5)))
                    print('I_ep {}, train {} times, reward:{}'.format(i_ep, self.training_step, np.round(rew, 5)))

                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]) # 3 dim
                # action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                # ratio = (action_prob / old_action_log_prob[index]) # TODO verify this method

                ratio = torch_RMSE(action_prob) / torch_RMSE(old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience
