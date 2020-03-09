import pickle
import torch.nn as nn
import torch as T
from utils import BCO
import numpy as np
from utils import DS_Inv, DS_Policy, DS_Policy_2
from torch.utils.data import DataLoader
import argparse
import gym
from collections import namedtuple
from models_cont import PPO
from itertools import count

envname = 'Hopper-v2'

POLICY = 'mlp'
env = gym.envs.make(envname)
num_state = env.observation_space.shape[0]
num_action = env.action_space.shape[0]
agent_1 = PPO(num_state, num_action)
agent_2 = PPO(num_state, num_action)


Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

trajs_demo = pickle.load(open('./data/Hopper-v1.pkl', 'rb'))  #TODO
ld_demo = DataLoader(DS_Inv(trajs_demo), batch_size=100, shuffle=True)  # ob1, a, ob2

print(len(ld_demo))
for obs1, _, obs2 in ld_demo:
    print(obs1.shape, obs2.shape)
    break

loss_func = nn.MSELoss()
EPOCHS = 20
M = 10000
EPS = 0.9
DECAY = 0.7

model_1 = BCO(env, policy=POLICY, is_continuous=True)
model_2 = BCO(env, policy=POLICY, is_continuous=True)
optim_1 = T.optim.Adam(model_1.parameters(), lr=5e-3)
optim_2 = T.optim.Adam(model_2.parameters(), lr=5e-3)

# training
for e in range(EPOCHS):
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
        # inp = T.from_numpy(obs).view(((1,) + obs.shape))
        out = agent_1.select_action(obs)

        if np.random.rand() >= EPS:
            act = out.numpy()[0]
        else:
            act = env.action_space.sample()
        new_obs, r, done, _ = env.step(act)
        trajs_inv.append([np.array(obs, dtype='f'), np.array(new_obs, dtype='f'), np.array(act, dtype='f')])
        # trajs_inv.append([obs, new_obs, act])
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
        # tmp = act.numpy().astype(int)
        # b = np.zeros((tmp.shape[0], 2))
        # b[np.arange(len(tmp)), tmp] = 1
        # # b = np.zeros((len(act),2))
        # act = T.Tensor(b)
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
        out = model_1.pred_inv(obs1, obs2)  # action
        obs = obs1.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        prob = out.max(axis=1)
        out = np.argmax(out, axis=1)  #TODO
        for i in range(100):
            traj_policy.append([obs[i], out[i], prob[i], obs2.cpu().detach().numpy()[i]])
    # step4, update policy via demo samples
    ld_policy = DataLoader(DS_Policy_2(traj_policy), batch_size=1000, shuffle=True) # [obs, act, p, ob2]

    cnt = 0
    del agent_1.buffer[:]

    # obs1, obs2 from demo; act(dim 1), action_prob(dim 1) from idm
    for obs, act, action_prob, obs2 in ld_policy:
        cnt += 1
        if cnt == 2:
            break
        for i in range(len(obs)):
            reward = np.square(
                float(act[i]) - float(model_2.pred_inv(obs[i].reshape(1, -1), obs2[i].reshape(1, -1))[0][0]))  # TODO
            trans = Transition(obs[i].numpy(), act[i].int(), action_prob[i].float(), reward, obs2[i].numpy())
            agent_1.store_transition(trans)
    # print('length of buffer: ', len(agent.buffer))
    agent_1.update(cnt)

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
        # inp = T.from_numpy(obs).view(((1,) + obs.shape))
        out = agent_2.select_action(obs)

        if np.random.rand() >= EPS:
            act = out.numpy()[0]
        else:
            act = env.action_space.sample()
        new_obs, r, done, _ = env.step(act)
        trajs_inv.append([np.array(obs, dtype='f'), np.array(new_obs, dtype='f'), np.array(act, dtype='f')])
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
        # tmp = act.numpy().astype(int)
        # b = np.zeros((tmp.shape[0], 2))
        # b[np.arange(len(tmp)), tmp] = 1
        # # b = np.zeros((len(act),2))
        # act = T.Tensor(b)
        ls_bh = loss_func(out, act.float())
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
        prob = out.max(axis=1)
        out = np.argmax(out, axis=1)
        for i in range(100):
            traj_policy.append([obs[i], out[i], prob[i], obs2.cpu().detach().numpy()[i]])
    # step4, update policy via demo samples
    ld_policy = DataLoader(DS_Policy_2(traj_policy), batch_size=1000, shuffle=True)

    cnt = 0
    del agent_2.buffer[:]
    for obs, act, action_prob, obs2 in ld_policy:
        cnt += 1
        if cnt == 2:
            break
        for i in range(len(obs)):
            reward = np.square(
                float(act[i]) - float(model_1.pred_inv(obs[i].reshape(1, -1), obs2[i].reshape(1, -1))[0][0]))
            trans = Transition(obs[i].numpy(), act[i].int(), action_prob[i].float(), reward, obs2[i].numpy())
            agent_2.store_transition(trans)
    # print('length of buffer: ', len(agent.buffer))
    agent_2.update(cnt)

    print('Ep %d: reward=%.2f, loss_inv=%.3f' % (e + 1, rews, ls_ep / len(ld_inv)))
    EPS *= DECAY

# evaluation
eps = 20
rews = np.zeros(eps)
trals = np.zeros(eps)
random = 1
log_states = []
log_rewards = []
log_acts = []
for i in range(eps):
    logs_s = []
    logs_r = []
    logs_a = []
    rew_tol = 0
    # env.seed(0)
    obs = env.reset()
    # obs = trajs_demo[0][0][0]
    done = False
    tra_l = 0
    while done == False:
        logs_s.append(obs)
        if random:
            act = env.action_space.sample()
        else:
            out = agent_2.select_action(obs)
            act = out
        obs, rew_one_step, done, _ = env.step(act)
        # print(done)
        rew_tol += rew_one_step
        tra_l += 1
        logs_r.append(rew_one_step)
        logs_a.append([act, out])
    rews[i] = rew_tol
    trals[i] = tra_l
    log_states.append(logs_s)
    log_rewards.append(logs_r)
    log_acts.append(logs_a)
    print(rew_tol, tra_l)
print(rews.mean(), rews.std(), rews.max())

# change environment parameters and check the model performance under the different environments
rew = []
for k in range(10):
    new_env = gym.make('Hopper-v2')
    mb = new_env.env.model.body_mass
    mb = np.array(mb)
    np.random.seed(19)
    mb = mb + np.random.random(mb.shape) * k

    new_env.env.model.body_mass = mb

    # evaluation
    eps = 20
    rews = np.zeros(eps)
    trals = np.zeros(eps)
    random = 0
    log_states = []
    log_rewards = []
    log_acts = []
    for i in range(eps):
        logs_s = []
        logs_r = []
        logs_a = []
        rew_tol = 0
        # env.seed(0)
        obs = new_env.reset()
        # obs = trajs_demo[0][0][0]
        done = False
        tra_l = 0
        while done == False:
            logs_s.append(obs)
            if random:
                act = new_env.action_space.sample()
            else:
                out = agent_1.select_action(obs)

                act = out
            obs, rew_one_step, done, _ = new_env.step(act)
            # print(done)
            rew_tol += rew_one_step
            tra_l += 1
            logs_r.append(rew_one_step)
            logs_a.append([act, out])
        rews[i] = rew_tol
        trals[i] = tra_l
        log_states.append(logs_s)
        log_rewards.append(logs_r)
        log_acts.append(logs_a)
        # print(rew_tol, tra_l)
    print('agent 1', rews.mean(), rews.std(), rews.max())
    rew.append(rews.mean())

import pickle
with open('./results/'+envname + '_rilo-res-friction.pkl', 'wb') as fo:
    pickle.dump(rew, fo)