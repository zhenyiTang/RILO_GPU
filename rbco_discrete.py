
import pickle
import torch.nn as nn
import torch as T
from bcoClass import BCO
import numpy as np
from data import load_demonstration, DS_Inv, DS_Policy
from torch.utils.data import DataLoader
import argparse
import gym

parser = argparse.ArgumentParser()
parser.add_argument('--envname', default='CartPole-v1')
parser.add_argument('--POLICY', default='mlp')
args = parser.parse_args()
POLICY = args.POLICY
env = gym.envs.make(args.envname)

envname = args.envname

trajs_demo = load_demonstration(args.envname)  # len = 50000
ld_demo = DataLoader(DS_Inv(trajs_demo), batch_size=100, shuffle=True)  # ob1, a, ob2

print(len(ld_demo))
for obs1, _, obs2 in ld_demo:
    print(obs1.shape, obs2.shape)
    break

loss_func = nn.MSELoss()
EPOCHS = 20
M = 5000
EPS = 0.9
DECAY = 0.7

model_1 = BCO(env)
model_2 = BCO(env)
optim_1 = T.optim.Adam(model_1.parameters(), lr=5e-3)
optim_2 = T.optim.Adam(model_2.parameters(), lr=5e-3)

for e in  range(EPOCHS):
    # step1, generate inverse samples
    epn = 0
    rews = 0
    trajs_inv = []
    done = False
    obs = env.reset()
    if e == 0:
        I = int(M / 0.02)
        I = M
    else:
        I = M
    for cnt in range(I):
        if done:
            obs = env.reset()
        inp = T.from_numpy(obs).view(((1,) + obs.shape))
        out = model_1.pred_act(inp.float()).cpu().detach().numpy()
        if np.random.rand() >= EPS:
            act = np.argmax(out, axis=1)[0]
        else:
            act = env.action_space.sample()
        new_obs, r, done, _ = env.step(act)
        trajs_inv.append([obs, new_obs, act])
        obs = new_obs
        rews += r
        if done == True:
            epn += 1
    rews /= epn
    # step2, update inverse model
    ld_inv = DataLoader(DS_Inv(trajs_inv), batch_size=32, shuffle=True)
    ls_ep = 0
    for obs1, obs2, act in ld_inv:
        out = model_1.pred_inv(obs1, obs2)
        tmp = act.numpy().astype(int)
        b = np.zeros((tmp.shape[0], 2))
        b[np.arange(len(tmp)), tmp] = 1
        # b = np.zeros((len(act),2))
        act = T.Tensor(b)
        ls_bh = loss_func(out, act.float())
        optim_1.zero_grad()
        ls_bh.backward()
        optim_1.step()
        ls_bh = ls_bh.cpu().detach().numpy()
        #        TQ.set_postfix(loss_inv='%.3f' % (ls_bh))
        ls_ep += ls_bh
    # step3, predict inverse action for demo samples
    traj_policy = []
    idx = np.random.randint(len(ld_demo))
    for obs1, _, obs2 in ld_demo:
        obs1 = obs1.double()
        obs2 = obs2.double()
        out = model_1.pred_inv(obs1, obs2)
        obs = obs1.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        out = np.argmax(out, axis=1)
        for i in range(100):
            traj_policy.append([obs[i], out[i]])
    # step4, update policy via demo samples
    ld_policy = DataLoader(DS_Policy(traj_policy), batch_size=32, shuffle=True)

    ls_eps = 0
    for obs, act in ld_policy:
        out = model_1.pred_act(obs.float())
        out_2 = model_2.pred_act(obs.float())
        tmp = act.numpy().astype(int)
        b = np.zeros((tmp.shape[0], 2))
        b[np.arange(len(tmp)), tmp] = 1
        act = T.Tensor(b)
        ls_bh = loss_func(out, act.float()) - loss_func(out_2, act.float())
        optim_1.zero_grad()
        ls_bh.backward()
        optim_1.step()
        ls_bh = ls_bh.cpu().detach().numpy()

        ls_eps += ls_bh
    ls_eps /= len(ld_policy)

    print('Ep %d: reward=%.2f, loss_inv=%.3f, loss_policy=%.3f' % (e + 1, rews, ls_ep / len(ld_inv), ls_eps))
    ######################################################################################################

    # step1, generate inverse samples
    epn = 0
    rews = 0
    trajs_inv = []
    done = False
    obs = env.reset()
    if e == 0:
        I = int(M / 0.02)
        I = M
    else:
        I = M
    for cnt in range(I):
        if done:
            obs = env.reset()
        inp = T.from_numpy(obs).view(((1,) + obs.shape))
        out = model_2.pred_act(inp.float()).cpu().detach().numpy()
        if np.random.rand() >= EPS:
            act = np.argmax(out, axis=1)[0]
        else:
            act = env.action_space.sample()
        new_obs, r, done, _ = env.step(act)
        trajs_inv.append([obs, new_obs, act])
        obs = new_obs
        rews += r
        if done == True:
            epn += 1
    rews /= epn
    # step2, update inverse model
    ld_inv = DataLoader(DS_Inv(trajs_inv), batch_size=32, shuffle=True)
    ls_ep = 0
    for obs1, obs2, act in ld_inv:
        out = model_2.pred_inv(obs1, obs2)
        tmp = act.numpy().astype(int)
        b = np.zeros((tmp.shape[0], 2))
        b[np.arange(len(tmp)), tmp] = 1
        # b = np.zeros((len(act),2))
        act = T.Tensor(b)
        ls_bh = -loss_func(out, act.float())
        optim_2.zero_grad()
        ls_bh.backward()
        optim_2.step()
        ls_bh = ls_bh.cpu().detach().numpy()
        #        TQ.set_postfix(loss_inv='%.3f' % (ls_bh))
        ls_ep += ls_bh
    # step3, predict inverse action for demo samples
    traj_policy = []
    idx = np.random.randint(len(ld_demo))
    for obs1, _, obs2 in ld_demo:
        obs1 = obs1.double()
        obs2 = obs2.double()
        out = model_2.pred_inv(obs1, obs2)
        obs = obs1.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        out = np.argmax(out, axis=1)
        for i in range(100):
            traj_policy.append([obs[i], out[i]])
    # step4, update policy via demo samples
    ld_policy = DataLoader(DS_Policy(traj_policy), batch_size=32, shuffle=True)

    ls_eps = 0
    for obs, act in ld_policy:
        out = model_2.pred_act(obs.float())
        tmp = act.numpy().astype(int)
        b = np.zeros((tmp.shape[0], 2))
        b[np.arange(len(tmp)), tmp] = 1
        act = T.Tensor(b)
        ls_bh = -loss_func(out, act.float())
        optim_2.zero_grad()
        ls_bh.backward()
        optim_2.step()
        ls_bh = ls_bh.cpu().detach().numpy()

        ls_eps += ls_bh
    ls_eps /= len(ld_policy)

    print('Ep %d: reward=%.2f, loss_inv=%.3f, loss_policy=%.3f' % (e + 1, rews, ls_ep / len(ld_inv), ls_eps))

    # step5, save model
    EPS *= DECAY
