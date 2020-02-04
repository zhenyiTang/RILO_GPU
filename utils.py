import numpy as np
from torch.utils.data import Dataset

import torch as T
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
# style

class BCO(nn.Module):
    def __init__(self, env, is_continuous, policy='mlp'):
        super(BCO, self).__init__()
        self.is_continuous = is_continuous
        self.policy = policy
        if is_continuous:
            self.act_n = env.action_space.shape[0]
        else:
            self.act_n = env.action_space.n

        if self.policy == 'mlp':
            self.obs_n = env.observation_space.shape[0]
            self.pol = nn.Sequential(*[nn.Linear(self.obs_n, 32), nn.LeakyReLU(),
                                       nn.Linear(32, 32), nn.LeakyReLU(),
                                       nn.Linear(32, self.act_n)])
            self.inv = nn.Sequential(*[nn.Linear(self.obs_n * 2, 32), nn.LeakyReLU(),
                                       nn.Linear(32, 32), nn.LeakyReLU(),
                                       nn.Linear(32, self.act_n)])

        elif self.policy == 'cnn':
            pass

    def pred_act(self, obs):
        out = self.pol(obs)
        if self.is_continuous:
            return out
        else:
            return out

    def pred_inv(self, obs1, obs2):
        obs = T.cat([obs1, obs2], dim=1)

        out = self.inv(obs.float())

        return out



# dataset used for inverse dynamic model
class DS_Inv(Dataset):
    def __init__(self, trajs):
        self.dataset = []


        for i, data in enumerate(trajs):
            # if i>5000:
            #     break
            obs, act, new_obs = data
            self.dataset.append([obs, act, new_obs])
        # for traj in trajs:
        #     for data in traj:
        #         obs, act, new_obs = data
        #         self.dataset.append([obs, act, new_obs])

    def __len__(self):
        return len(self.dataset)

    # WTY
    def __getitem__(self, idx):
        obs, act, new_obs = self.dataset[idx]

        # TODO
        return obs, act, new_obs
class DS_Policy_2(Dataset):
    def __init__(self, traj):
        self.dat = []

        for dat in traj:
            obs, act,p,ob2 = dat

            self.dat.append([obs, act,p,ob2])

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        obs, act,p,ob2 = self.dat[idx]

        return obs, np.asarray(act),p,ob2


# dataset used for policy
class DS_Policy(Dataset):
    def __init__(self, traj):
        self.dataset = []

        for dat in traj:
            obs, act = dat

            self.dataset.append([obs, act])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        obs, act = self.dataset[idx]

        return obs, np.asarray(act)

import torch


from constants import *

def get_gaussian_log(x, mu, log_stddev):
    '''
    returns log probability of picking x
    from a gaussian distribution N(mu, stddev)
    '''
    # ignore constant since it will be cancelled while taking ratios
    log_prob = -log_stddev - (x - mu) ** 2 / (2 * torch.exp(log_stddev) ** 2)
    return log_prob

def plot_data(statistics):
    '''
    plots reward and loss graph for entire training
    '''
    x_axis = np.linspace(0, N_EPISODES, N_EPISODES // LOG_STEPS)
    plt.plot(x_axis, statistics["reward"])
    plt.title("Variation of mean rewards")
    plt.show()

    plt.plot(x_axis, statistics["val_loss"])
    plt.title("Variation of Critic Loss")
    plt.show()

    plt.plot(x_axis, statistics["policy_loss"])
    plt.title("Variation of Actor loss")
    plt.show()

def curve_drawing(x,y1,y2,path):
    # x = range(0, 10)
    # y1 = np.array([175.61, 177.52, 184.34, 188.25,
    #                196.35, 199.13, 201.3, 205.51, 206.94, 207.96
    #                ])
    # y1 = np.array([270.9216132726946, 312.05857291249106, 98.74493011472858, 259.45012155174885, 263.7087507131492,
    #                238.45949074221681, 198.95666299552016, 204.33321927094272, 190.84042375287873, 78.01057217510491]
    #               )
    # y2 = np.array([255.64, 255.35, 251.98, 247.34, 211.49,
    #                145.5, 129.09, 131.39, 139.6, 129.9
    #                ])
    # y2 = np.array([175.90775451660156, 179.34550399780272, 183.20120544433593, 187.32345886230468, 191.8228759765625,
    #                196.67101440429687, 200.206982421875, 201.01299819946288, 204.7893913269043, 206.10998992919923]
    #               )
    # multiple line plot
    #plt.style.use('seaborn-darkgrid')
    plt.plot(x, y1, marker='', color='red', linewidth=1, alpha=0.9, label='RAILfO')
    plt.plot(x, y2, marker='', color='green', linewidth=1, alpha=0.9, label='BCO')
    # Add legend
    plt.legend(ncol=2, prop={'size': 12})

    # Add titles
    # plt.title("Performance under Different Friction", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Level of Gaussian Noise", fontsize= 14)
    plt.ylabel("Total Rewards", fontsize= 14)
    plt.show()
    plt.savefig(path)
    #plt.savefig('C:/Users/xueh2/Box/mass.png')
