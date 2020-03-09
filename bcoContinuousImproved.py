import gym
import pickle
import numpy as np
import torch as T
from torch.utils.data import DataLoader
from bcoClass import BCO
from utils import DS_Inv, DS_Policy

#  discrete

# init the model
envs = ['Acrobot-v1', 'MountainCar-v0', 'CartPole-v1', 'Reacher-v1', 'Hopper-v2']
envname = envs[-1]
env = gym.make(envname)
env.seed(42)
model = BCO(env)

# number of episodes
episodes = 200

# the number of transitions we use to train
# the inverse dynamic model over one episode
transition_max = 10000

# load trajs from the expert
demo_batch_size = 100
# trajs_demo = pickle.load(open('./data/' + envname + '.pkl', 'rb'))  #pkl: 50000 * (state1(11 dim), action(3dim), state2(11dim))
trajs_demo = pickle.load(open('./data/Hopper-v2.pkl', 'rb'))
ld_demo = DataLoader(DS_Inv(trajs_demo), batch_size=demo_batch_size)


# the trajs got from policy model and used to update inverse model
# for e in tqdm(range(episodes)):
# after for loop, we will have a good policy model from idm
trajs_inv = [] # TZY: trajs_inv is ground truth data, which shouldn't be wiped out every time in new episodes
for e in range(episodes):

    ######## step1, generate inverse samples using random and policy mix ########
    # number of state-action-next_state tuple (transition)
    trans_num = 0
    # store the num of total trajs
    traj_num = 0
    # total rewards of all trajs
    rew_tot = 0

    # collect trajs_inv{s_i, a_i, s_i+1}
    while True:
        obs = env.reset() # init obs
        while True:
            # epsilon greedy policy
            if np.random.rand() >= model.epsilon:
                # notice that obs.shape is (some, )
                inp = T.from_numpy(obs).view(((1,) + obs.shape)).float()
                out = model.pred_act_by_policy(inp).cpu().detach().numpy()
                act = out[0]  #act is 3d
            else:
                act = env.action_space.sample()
            # get the next_obs
            next_obs, rew_one_step, done, _ = env.step(act)
            env.render()  # TZY : only render for now
            # regular update
            trajs_inv.append([np.array(obs, dtype='f'), np.array(act, dtype='f'),
                              np.array(next_obs, dtype='f')])
            rew_tot += rew_one_step

            obs = next_obs

            trans_num += 1
            if done:
                traj_num += 1
                break
        # the length of total transitions cannot be larger than M
        if trans_num >= transition_max:
            break

    # get the average reward over all the trajs
    rew_avg = rew_tot / traj_num
    print("line 78: rew_avg : ", rew_avg, " rew_total : ", rew_tot)

    ######## step2, update inverse model by Step 1 traj ########
    ld_inv = DataLoader(DS_Inv(trajs_inv), batch_size=32, shuffle=True)
    # with tqdm(ld_inv) as TQ:
    # total loss of an episode
    ls_tot_inv = 0
    for obs1, act, obs2 in ld_inv:
        out = model.pred_act_by_inv(obs1.float(), obs2.float())
        # store the loss of each batch

        ls_bh = model.loss_func(out, act)
        ls_tot_inv += ls_bh.cpu().detach().numpy()
        # update the inverse dynamic model
        #for param in model.parameters():
        #    print(param.data)
        model.model_update(ls_bh)
        # average loss

    ######## step3, predict inverse action for demo samples ########
    # use the inverse dynamic model to produce samples with actions
    # from samples without actions for policy model updates. And NOTICE
    # that we do not need next_state to train the policy network
    traj_policy = []
    # with tqdm(ld_demo) as TQ:
    # we do not access the actions in the trajs
    for obs1, _, obs2 in ld_demo:
        # lose the "useless" next_state
        obs = obs1.cpu().detach().numpy()
        out = model.pred_act_by_inv(obs1.float(), obs2.float())
        out = out.cpu().detach().numpy()

        for i in range(demo_batch_size):
            traj_policy.append([obs[i], out[i]])

    ######## step4, update policy via demo samples ########
    ld_policy = DataLoader(DS_Policy(traj_policy), batch_size=32, shuffle=True)
    # with tqdm(ld_policy) as TQ:
    # the total lost of an episode
    ls_tot = 0
    for obs, act in ld_policy:
        out = model.pred_act_by_policy(obs.float())
        # store the loss of each batch
        ls_bh = model.loss_func(out, act)
        ls_tot += ls_bh.cpu().detach().numpy()
        # update the policy model
        model.model_update(ls_bh)
    ls_avg = ls_tot / len(ld_policy)
    print('Ep %d: reward=%.2f,'
          'loss_inv=%.3f,loss_policy=%.3f '
          % (e + 1, rew_avg, ls_tot_inv / len(ld_inv), ls_avg))

    ######## step5, save model ########
    # T.save(model.state_dict(), 'model/model_cart-pole_%d.pt' % (e + 1))
    model.lower_exploration_rate()


# change environment parameters and check the model performance under the different environments
rew = []
random = 0
for k in range(10):
    # change environment friction
    new_env = gym.make(envname)

    mb = new_env.env.model.geom_friction
    mb = np.array(mb)
    np.random.seed(19)
    mb = mb + np.random.random(mb.shape) * k
    # new_env.env.model.geom_friction = mb  #TODO 'geom_friction' of 'mujoco_py.cymj.PyMjModel' objects is not writable

    # evaluation
    eps = 20
    rews = np.zeros(eps)
    trals = np.zeros(eps)
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
            act = model.pred_act_by_policy(T.tensor(obs).float()).cpu().detach().numpy()
            obs, rew_one_step, done, _ = new_env.step(act)
            new_env.render()  # TZY : only render for now
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
    print(rews.mean(), rews.std(), rews.max())
    rew.append(rews.mean())

import pickle

with open('./results/'+envname + '_bco-res-friction.pkl', 'wb') as fo:
    pickle.dump(rew, fo)