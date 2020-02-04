import gym
import pickle
import numpy as np
import torch as T
from torch.utils.data import DataLoader
from bcoClass import BCO
from utils import DS_Inv, DS_Policy,DS_Policy_2
from models import PPO
from collections import namedtuple

#  discrete
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

# init the model
env = gym.make('CartPole-v1')
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
agent = PPO(num_state, num_action)
model = BCO(env)

# number of episodes
episodes = 25

# the number of transitions we use to train
# the inverse dynamic model over one episode
transition_max = 1000

# load trajs from the expert
demo_batch_size = 100
trajs_demo = pickle.load(open('./data/CartPole-v1.pkl', 'rb'))
ld_demo = DataLoader(DS_Inv(trajs_demo), batch_size=demo_batch_size)

# the trajs got from policy model and used to update inverse model


# for e in tqdm(range(episodes)):
for e in range(episodes):
    trajs_inv = []
    ######## step1, generate inverse samples ########
    # number of state-action-next_state tuple (transition)
    trans_num = 0
    # store the num of total trajs
    traj_num = 0
    # total rewards of all trajs
    rew_tot = 0
    while True:
        # store the transitions of a traj
        # traj = []
        # store the total of a single traj
        # rew_traj = 0
        obs = env.reset()
        while True:
            # notice that obs.shape is (some, )
            inp = T.from_numpy(obs).view(((1,) + obs.shape)).float()
            out, action_prob = agent.select_action(obs)
            #out = model.pred_act_by_policy(inp).cpu().detach().numpy()
            # epsilon greedy policy
            if np.random.rand() >= model.epsilon:
                act = out
            else:
                act = env.action_space.sample()
            # get the next_obs
            next_obs, rew_one_step, done, _ = env.step(act)
            # regular update
            trajs_inv.append([obs, act, next_obs])
            obs = next_obs
            rew_tot += rew_one_step
            trans_num += 1
            if done:
                traj_num += 1
                break
        # the length of total transitions cannot be larger than M
        if trans_num >= transition_max:
            break

    # get the average reward over all the trajs
    rew_avg = rew_tot / traj_num

    ######## step2, update inverse model ########
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
        # obs1 = obs1.double()
        # obs2 = obs2.double()
        out = model.pred_act_by_inv(obs1, obs2)
        obs = obs1.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        prob = out.max(axis=1)
        out = np.argmax(out, axis=1)
        # # lose the "useless" next_state
        # obs = obs1.cpu().detach().numpy()
        # out = model.pred_act_by_inv(obs1.float(), obs2.float())
        # out = out.cpu().detach().numpy()
        # out = np.argmax(out, axis=1)
        for i in range(demo_batch_size):
            traj_policy.append([obs[i], out[i], prob[i], obs2.cpu().detach().numpy()[i]])
            #traj_policy.append([obs[i], out[i]])

    ######## step4, update policy via demo samples ########
    ld_policy = DataLoader(DS_Policy_2(traj_policy), batch_size=32, shuffle=True)

    cnt = 0
    del agent.buffer[:]
    for obs, act, action_prob, obs2 in ld_policy:
        cnt += 1
        if cnt == 2:
            break
        for i in range(len(obs)):
            reward = np.square(
                float(act[i]) - float(model.pred_act_by_inv(obs[i].reshape(1, -1), obs2[i].reshape(1, -1))[0][0]))
            trans = Transition(obs[i].numpy(), act[i].int(), action_prob[i].float(), reward, obs2[i].numpy())
            agent.store_transition(trans)
    # print('length of buffer: ', len(agent.buffer))
    agent.update(cnt)

    print('Ep %d: reward=%.2f,'
          'loss_inv=%.3f'
          % (e + 1, rew_avg, ls_tot_inv / len(ld_inv)))

    model.lower_exploration_rate()

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
    obs = env.reset()
    # obs = trajs_demo[0][0][0]
    done = False
    tra_l = 0
    while done == False:
        logs_s.append(obs)
        if random:
            act = env.action_space.sample()
        else:
            out = model.pred_act_by_policy(T.tensor(obs).float()).cpu().detach().numpy()
            act = np.argmax(out)
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
