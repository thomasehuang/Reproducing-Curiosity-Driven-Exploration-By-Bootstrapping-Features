import os
import datetime
import argparse
import gym
import tensorflow as tf
from embedding import *
from policy import *
from forward_dynamics import *

from atari_wrappers import wrap_deepmind
from ppo import PPO
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='gym environment ID', default='Pong-v0')
parser.add_argument('--seed', help='seed for environment', type=int, default=0)
parser.add_argument('--num-timesteps', help='number of timesteps', type=int, default=int(1e5))
parser.add_argument('--backprop-to-embedding', help='is backprop to embedding', type=bool, default=False)


def save_to_file(directory, graph_rewards, graph_epi_lens, graph_in_rewards):
    with open(directory + '/pong_frf_rewards.txt', 'a+') as reward_file:
        for graph_reward, timestep in graph_rewards:
            reward_file.write("%s %s\n" % (graph_reward, timestep))
        graph_rewards[:] = []
    with open(directory + '/pong_frf_ep_len.txt', 'a+') as ep_len_file:
        for graph_epi_len, timestep in graph_epi_lens:
            ep_len_file.write("%s %s\n" % (graph_epi_len, timestep))
        graph_epi_lens[:] = []
    with open(directory + '/pong_frf_in_rewards.txt', 'a+') as in_reward_file:
        for graph_in_reward, timestep in graph_in_rewards:
            in_reward_file.write("%s %s\n" % (graph_in_reward, timestep))
        graph_in_rewards[:] = []


def cbf(env, sess,
        replay_size, # size of replay buffer
        batch_size, # size of minibatch
        n_timesteps, # number of timesteps
        len_rollouts, # length of each rollout
        n_optimizations, # number of optimization steps
        embedding_space_size, # size of embeddings
        learning_rate, # learning rate
        is_backprop_to_embedding=False):

    # Initialize models
    emb = CnnEmbedding("embedding", env.observation_space, env.action_space, embedding_space_size)
    fd = ForwardDynamics("forward_dynamics", embedding_space_size, env.action_space)
    policy = Policy("policy_new", env.action_space, is_backprop_to_embedding, emb=emb, emb_space=embedding_space_size)
    policy_old = Policy("policy_old", env.action_space, is_backprop_to_embedding, emb=emb, emb_space=embedding_space_size)
    ppo = PPO(env, policy, policy_old,
              max_timesteps=int(n_timesteps * 1.1),
              timesteps_per_actorbatch=256,
              clip_param=0.2, entcoeff=0.01,
              optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
              gamma=0.99, lam=0.95,
              schedule='linear',
              is_backprop_to_embedding=is_backprop_to_embedding,
    )

    n_rollouts = n_timesteps // len_rollouts

    replay_memory = ReplayMemory(replay_size)
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

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
    if is_backprop_to_embedding:
        s_arr = np.array([np.zeros([84,84,4]) for _ in range(len_rollouts)])
    else:
        s_arr = np.array([np.zeros(512) for _ in range(len_rollouts)])
    r_arr = np.zeros(len_rollouts, 'float32')
    vpreds = np.zeros(len_rollouts, 'float32')
    dones = np.zeros(len_rollouts, 'int32')
    a_arr = np.array([a for _ in range(len_rollouts)])

    # For graphing
    best_reward = -21
    cur_reward = 0
    graph_rewards = []
    graph_epi_lens = []
    graph_in_rewards = []

    cur_time = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    directory = 'results/' + cur_time
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(n_rollouts):
        # print('# rollout: %i. timestep: %i' % (i,t,))
        for j in range(len_rollouts):
            if t > 0 and t % int(1e3) == 0:
                # print('# frame: %i. Best reward so far: %i.' % (t, best_reward,))
                save_to_file(directory, graph_rewards, graph_epi_lens, graph_in_rewards)

                save_path = saver.save(sess, "model/model.ckpt")
                #print("Model saved in file: %s" % save_path)

            s = np.array(s)
            obs1 = emb.embed([s])
            if is_backprop_to_embedding:
                a, vpred = policy.act([s])
            else:
                a, vpred = policy.act(obs1)

            # update optimization batch variables
            idx = t % len_rollouts
            if is_backprop_to_embedding:
                s_arr[idx] = s
            else:
                s_arr[idx] = obs1
            vpreds[idx] = vpred
            dones[idx] = done
            a_arr[idx] = a

            s_ , ext_r, done, _ = env.step(a)

            cur_reward += ext_r
            # graph_rewards.append(best_reward)

            s_ = np.array(s_)

            # compute intrinsic reward
            obs2 = emb.embed([s_])
            r = fd.get_loss(obs1, obs2, np.eye(env.action_space.n)[a])
            # print(r)
            replay_memory.add((s, a, r, s_))
            if t > 0 and t % int(2e2) == 0:
                graph_in_rewards.append((r, t))

            # update optimization batch variables
            r_arr[idx] = r
            cur_ep_ret += r
            cur_ep_len += 1

            # Prepare for next step
            if done:
                graph_epi_lens.append((cur_ep_len,t))
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                graph_rewards.append((best_reward, t))
                cur_reward = 0
                s = env.reset()
            else:
                s = s_
            t += 1
        for j in range(n_optimizations):
            # optimize theta_pi (and optionally theta_phi) wrt PPO loss
            ppo.step({"ob" : s_arr, "rew" : r_arr, "vpred" : vpreds, "new" : dones,
                      "ac" : a_arr, "nextvpred": vpred * (1 - done),
                      "ep_rets" : ep_rets, "ep_lens" : ep_lens})
            # sample minibatch M from replay buffer R
            states, actions, rewards, next_states = replay_memory.sample(batch_size)
            obs1, obs2 = emb.embed(states), emb.embed(next_states) # embedding of states
            actions = np.squeeze([np.eye(env.action_space.n)[action] for action in actions])
            # optimize theta_f wtf forward dynamics loss on minibatch
            fd.train(obs1, obs2, actions, learning_rate)
            # optionally optimize theta_phi, theta_A wrt to auxilary loss

    save_to_file(directory, graph_rewards, graph_epi_lens, graph_in_rewards)


def main():
    args = parser.parse_args()
    with tf.Session() as sess:
        env = wrap_deepmind(gym.make(args.env), episode_life=False, clip_rewards=False, frame_stack=True)
        env.seed(args.seed)

        cbf(env, sess,
            replay_size=1000,
            batch_size=128,
            n_timesteps=args.num_timesteps,
            len_rollouts=64,
            n_optimizations=8,
            embedding_space_size=288,
            learning_rate=1e-3,
            is_backprop_to_embedding=args.backprop_to_embedding
            )


if __name__ == '__main__':
    main()
