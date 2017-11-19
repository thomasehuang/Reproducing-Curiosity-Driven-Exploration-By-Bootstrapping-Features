# TODO integrate PPO
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from embedding import *
from policy import *
from forward_dynamics import *

from atari_wrappers import wrap_deepmind
from ppo import PPO
from replay_memory import ReplayMemory


def cbf(env,
        n_rollouts,
        len_rollouts,
        n_optimizations,
        embedding_space_size,
        learning_rate,
        is_backprop_to_embedding=False):
    # Initialize models
    emb = CnnEmbedding("embedding", env.observation_space, env.action_space, embedding_space_size)
    fd = ForwardDynamics("forward_dynamics", embedding_space_size, env.action_space)
    policy = Policy("policy_new", env.action_space, is_backprop_to_embedding, emb=emb, emb_space=embedding_space_size)
    policy_old = Policy("policy_old", env.action_space, is_backprop_to_embedding, emb=emb, emb_space=embedding_space_size)
    ppo = PPO(env, policy, policy_old,
              max_timesteps=int(int(10e6) * 1.1),
              timesteps_per_actorbatch=256,
              clip_param=0.2, entcoeff=0.01,
              optim_epochs=8, optim_stepsize=1e-3, optim_batchsize=64,
              gamma=0.99, lam=0.95,
              schedule='linear',
              is_backprop_to_embedding=is_backprop_to_embedding,
    )

    replay_memory = ReplayMemory(REPLAY_SIZE)
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

    t = 0

    # initialize optimization batch variables
    a = env.action_space.sample() # not used, just so we have the datatype
    done = True # marks if we're on first timestep of an episode
    s = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    s_arr = np.array([np.zeros(512) for _ in range(len_rollouts)])
    r_arr = np.zeros(len_rollouts, 'float32')
    vpreds = np.zeros(len_rollouts, 'float32')
    dones = np.zeros(len_rollouts, 'int32')
    a_arr = np.array([a for _ in range(len_rollouts)])

    # For graphing
    graph_rewards = []
    best_reward = -21
    cur_reward = 0

    for i in range(n_rollouts):
        print('# rollout: %i. timestep: %i' % (i,t,))
        for j in range(len_rollouts):
            if t % int(1e3) == 0:
                print('# frame: %i. Best reward so far: %i.' % (t, best_reward,))

            s = np.array(s)
            obs1 = emb.embed([s])
            a, vpred = policy.act(obs1)

            # update optimization batch variables
            idx = t % len_rollouts
            s_arr[idx] = obs1
            vpreds[idx] = vpred
            dones[idx] = done
            a_arr[idx] = a

            s_ , ext_r, done, _ = env.step(a)

            cur_reward += ext_r
            graph_rewards.append(best_reward)

            s_ = np.array(s_)

            # compute intrinsic reward
            obs2 = emb.embed([s_])
            r = fd.get_loss(obs1, obs2, np.eye(env.action_space.n)[a])
            replay_memory.add((s, a, r, s_))

            # update optimization batch variables
            r_arr[idx] = r
            cur_ep_ret += r
            cur_ep_len += 1

            # Prepare for next step
            if done:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                if cur_reward > best_reward:
                    best_reward = cur_reward
                cur_reward = 0
                s = env.reset()
            else:
                s = s_
            t += 1
        for j in range(N_OPTIMIZATIONS):
            # optimize theta_pi (and optionally theta_phi) wrt PPO loss
            ppo.step({"ob" : s_arr, "rew" : r_arr, "vpred" : vpreds, "new" : dones,
                      "ac" : a_arr, "nextvpred": vpred * (1 - done),
                      "ep_rets" : ep_rets, "ep_lens" : ep_lens})
            # sample minibatch M from replay buffer R
            states, actions, rewards, next_states = replay_memory.sample(BATCH_SIZE)
            obs1, obs2 = emb.embed(states), emb.embed(next_states) # embedding of states
            actions = np.squeeze([np.eye(env.action_space.n)[action] for action in actions])
            # optimize theta_f wtf forward dynamics loss on minibatch
            fd.train(obs1, obs2, actions, learning_rate)
            # optionally optimize theta_phi, theta_A wrt to auxilary loss

    plt.xlabel('Training frames')
    plt.ylabel('Best return')
    plt.plot(range(1, len(graph_rewards)+1), graph_rewards, 'b--')
    plt.show()


N_ROLLOUTS = 1000
LEN_ROLLOUTS = 64
N_OPTIMIZATIONS = 10
EMBEDDING_SPACE_SIZE = 512
REPLAY_SIZE = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.5
IS_BACKPROP_TO_EMBEDDING = False

if __name__ == '__main__':
    with tf.Session() as sess:
        env = wrap_deepmind(gym.make('Pong-v0'), episode_life=False, clip_rewards=False, frame_stack=True)
        env.seed(42)
        cbf(env,
            N_ROLLOUTS,
            LEN_ROLLOUTS,
            N_OPTIMIZATIONS,
            EMBEDDING_SPACE_SIZE,
            LEARNING_RATE,
            IS_BACKPROP_TO_EMBEDDING
            )

