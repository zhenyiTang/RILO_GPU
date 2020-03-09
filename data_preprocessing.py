import os
from tqdm import tqdm_notebook as tqdm
import numpy as np
import tensorflow as tf
import pickle
import tf_util

import gym
from copy import deepcopy
import torch as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def load_policy(filename):  # pkl file
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    # data.keys() #['GaussianPolicy', 'nonlin_type': 'tanh']
    # data['GaussianPolicy']: its a NN consists of 3 layers: layer_0, layer_2 and ourpur layer
    # 'obsnorm' ['Standardizer'] # count = 35980906, meansq_1_D, mean_1_D shape(1,11),
    # 'logstdevs_1_Da' shape(1,3), array([[-0.99840918, -0.94493532, -1.52489462]])
    # 'out'['AffineLayer']: W shape(64,3), b shape(1,3) 'hidden']
    # ['hidden']['FeedforwardNet']['layer_0']['AffineLayer']['W'] shape(11, 64), b shape(1, 64)
    # ['hidden']['FeedforwardNet']['layer_2']['AffineLayer']['W'] shape(64, 64), b shape(1, 64)

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    # print nonlin_type
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

    # Keep track of input and output dims (i.e. observation and action dims) for the user

    def build_policy(obs_bo):
        def read_layer(l):
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

        def apply_nonlin(x):
            if nonlin_type == 'lrelu':
                return tf_util.lrelu(x, leak=.01)  # openai/imitation nn.py:233
            elif nonlin_type == 'tanh':
                return tf.tanh(x)
            else:
                raise NotImplementedError(nonlin_type)

        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        # print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
        normedobs_bo = (obs_bo - obsnorm_mean) / (
                obsnorm_stdev + 1e-6)  # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation

        curr_activations_bd = normedobs_bo

        # Hidden layers next
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            # print "\nW: " + str(W.shape)
            # print "b: " + str(b.shape)
            curr_activations_bd = apply_nonlin(tf.matmul(curr_activations_bd, W) + b)

        # Output layer
        W, b = read_layer(policy_params['out'])
        # print "\nOutput:"
        # print "W: " + str(W.shape)
        # print "b: " + str(b.shape)
        output_bo = tf.matmul(curr_activations_bd, W) + b
        return output_bo

    obs_bo = tf.placeholder(tf.float32, [None, None])
    a_ba = build_policy(obs_bo)  # input obs_bo, output a_ba
    policy_fn = tf_util.function([obs_bo], a_ba)
    return policy_fn


class Supervisor():
    def __init__(self, policy_fn, sess):
        self.policy_fn = policy_fn
        self.sess = sess
        with self.sess.as_default():
            tf_util.initialize()

    def sample_action(self, s):
        with self.sess.as_default():
            intended_action = self.policy_fn(s[None, :])[0]
            return intended_action

    def intended_action(self, s):
        return self.sample_action(s)


# generate trajectory
def generate_data(envname):
    """
        Preprocess hyperparameters and initialize learner and supervisor
    """
    parent_path = 'D:/AML/RBCO/'
    filename = 'D:/AML/RBCO/experts/' + envname + '.pkl'
    model_dir = 'D:/AML/RBCO/demonstration/' + envname + '/'
    env = gym.envs.make(envname)

    terminal = True
    sess = tf.Session()

    policy = load_policy(filename)  # return a policy function
    net_sup = Supervisor(policy, sess)  # sample_action = intended_action
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    state_state = []

    for itr in range(50000):
        if terminal:

            state = env.reset()
        old_state = deepcopy(state)
        a = net_sup.intended_action(old_state)
        state, reward, terminal, _ = env.step(a)
        # state_state.append(str(list(np.round(old_state, 12))) + '\t' + str(list(np.round(state, 12)))
        #                    + '\t' + str(list(np.round(a, 12))) + '\t' + str(np.round(reward, 12)) + '\n')
        state_state.append([np.array(old_state, dtype='f'),np.array(a, dtype='f'),
                           np.array(state, dtype='f')])
    with open('./data/'+envname+'.pkl', 'wb') as fo:
        pickle.dump(state_state, fo)
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
        obs = env.reset()
        # obs = trajs_demo[0][0][0]
        done = False
        tra_l = 0
        while done == False:
            act =net_sup.intended_action(obs)

            obs, rew_one_step, done, _ = env.step(act)
            # print(done)
            rew_tol += rew_one_step
            tra_l += 1
            logs_r.append(rew_one_step)
            logs_a.append([act, act])
        rews[i] = rew_tol
        trals[i] = tra_l
        log_states.append(logs_s)
        log_rewards.append(logs_r)
        log_acts.append(logs_a)
        print(rew_tol, tra_l)
    print(rews.mean(), rews.std(), rews.max())

generate_data('Hopper-v2')

# expert policy:
# reacher -4.1749+_1.9258 max = -0.8450
# hopper  3777.983013626925 3.982150030807179 3782.5077566927316
#ant 4813.365300987264 104.88047185949584 4990.760658125518
# humanoid 10405.64165857618 50.97695827688401 10515.74322905804
# halfcheetah 4127.18017476422 106.00016784765904 4292.629216720481
