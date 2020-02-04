import torch as T
import torch.nn as nn


class BCO(nn.Module):
    def __init__(self, env):
        super(BCO, self).__init__()
        try:
            self.nAct = env.action_space.shape[0]
            self.loss_func = nn.MSELoss()
        except:
            self.nAct = env.action_space.n
            self.loss_func = nn.CrossEntropyLoss()
        self.nObs = env.observation_space.shape[0]
        self.epsilon = 0.9
        self.decay_rate = 0.5
        # 0.9, 0.5, 5e-3
        self.build_policy_model()
        self.build_inverse_dynamic_model()
        self.optim = T.optim.Adam(self.parameters(), lr=7e-3)

    # the model learnt from imitaiton learning
    def build_policy_model(self):
        self.pol = nn.Sequential(*[
            nn.Linear(self.nObs, 32), nn.Tanh(),
            # nn.Linear(100, 32),nn.Tanh(), #nn.LeakyReLU(),
            # nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, self.nAct)
        ])

    # infer the missing action information
    def build_inverse_dynamic_model(self):
        self.inv = nn.Sequential(*[
            nn.Linear(self.nObs * 2, 32), nn.Tanh(),
            # nn.Linear(100, 32), nn.Tanh(), #nn.LeakyReLU(),
            # nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, self.nAct)
        ])

    def pred_act_by_policy(self, obs):
        return self.pol(obs)

    def pred_act_by_inv(self, obs1, obs2):
        # concatenate two obs to infer the missing action
        obs = T.cat([obs1, obs2], dim=1)
        return self.inv(obs)

    def lower_exploration_rate(self):
        self.epsilon *= self.decay_rate

    def model_update(self, loss_to_bp):
        self.optim.zero_grad()
        loss_to_bp.backward()
        self.optim.step()
