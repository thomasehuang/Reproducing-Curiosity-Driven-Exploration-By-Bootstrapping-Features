import os
import datetime
import argparse
import gym
import tensorflow as tf
from collections import deque
from embedding import *
from policy import *
from forward_dynamics import *
from inverse_dynamics import *

from atari_wrappers import wrap_deepmind, make_atari
from baselines import logger
from baselines.common import set_global_seeds
from ppo import PPO
from replay_buffer import ReplayBuffer
from mpi4py import MPI


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='gym environment ID', default='PongNoFrameskip-v4')
parser.add_argument('--seed', help='seed for environment', type=int, default=0)
parser.add_argument('--num-timesteps', help='number of timesteps', type=int, default=int(1e5))
parser.add_argument('--joint-training', help='is joint training', type=str2bool, default=False)
parser.add_argument('--using-extrinsic-reward', help='using extrinsic reward', type=str2bool, default=False)
parser.add_argument('--idf', help='run with idf', type=str2bool, default=False)
parser.add_argument('--inference', help='performing inference', type=str2bool, default=False)
parser.add_argument('--path-to-model', help='path to model', default='model/model.ckpt')
parser.add_argument('--debug', help='debug mode', type=str2bool, default=False)
parser.add_argument('--tensorboard', help='print to tensorboard', type=str2bool, default=False)


def save_to_file(directory, env_name, graph_rewards, graph_best_rewards, graph_epi_lens, graph_in_rewards, graph_avg_rewards, graph_avg_epi_lens):
    if len(graph_rewards) != 0:
        with open(directory + '/' + env_name + '_rewards.txt', 'a+') as reward_file:
            for graph_reward, timestep in graph_rewards:
                reward_file.write("%s %s\n" % (graph_reward, timestep))
            graph_rewards[:] = []
    if len(graph_best_rewards) != 0:
        with open(directory + '/' + env_name + '_best_rewards.txt', 'a+') as reward_file:
            for graph_best_reward, timestep in graph_best_rewards:
                reward_file.write("%s %s\n" % (graph_best_reward, timestep))
            graph_best_rewards[:] = []
    if len(graph_epi_lens) != 0:
        with open(directory + '/' + env_name + '_ep_len.txt', 'a+') as ep_len_file:
            for graph_epi_len, timestep in graph_epi_lens:
                ep_len_file.write("%s %s\n" % (graph_epi_len, timestep))
            graph_epi_lens[:] = []
    if len(graph_in_rewards) != 0:
        with open(directory + '/' + env_name + '_in_rewards.txt', 'a+') as in_reward_file:
            for graph_in_reward, timestep in graph_in_rewards:
                in_reward_file.write("%s %s\n" % (graph_in_reward, timestep))
            graph_in_rewards[:] = []
    if len(graph_avg_rewards) != 0:
        with open(directory + '/' + env_name + '_avg_rewards.txt', 'a+') as avg_reward_file:
            for graph_avg_reward, timestep in graph_avg_rewards:
                avg_reward_file.write("%s %s\n" % (graph_avg_reward, timestep))
            graph_avg_rewards[:] = []
    if len(graph_avg_epi_lens) != 0:
        with open(directory + '/' + env_name + '_avg_epi_lens.txt', 'a+') as avg_epi_len_file:
            for graph_avg_epi_len, timestep in graph_avg_epi_lens:
                avg_epi_len_file.write("%s %s\n" % (graph_avg_epi_len, timestep))
            graph_avg_epi_lens[:] = []


def inference(env, sess, env_name,
              path_to_model,
              embedding_space_size, # size of embeddings
              joint_training=False,
              using_extrinsic_reward=False
             ):
    # Initialize models
    emb = CnnEmbedding("embedding", env.observation_space, env.action_space, embedding_space_size)
    fd = ForwardDynamics("forward_dynamics", embedding_space_size, env.action_space) if not using_extrinsic_reward else None
    policy = Policy("policy_new", env.action_space, joint_training, emb_size=embedding_space_size, emb_network=emb)

    saver = tf.train.Saver()

    saver.restore(sess, path_to_model)

    s = env.reset()
    while True:
        env.render()

        s = np.array(s)
        if joint_training:
            a, _ = policy.act([s])
        else:
            obs1 = emb.embed([s])
            a, _ = policy.act(obs1)

        s_ , ext_r, done, _ = env.step(a)

        if done:
            s = env.reset()
        else:
            s = s_


def cbf(rank, env, sess, env_name, seed, debug,
        tensorboard, idf,
        replay_size, # size of replay buffer
        batch_size, # size of minibatch
        n_timesteps, # number of timesteps
        len_rollouts, # length of each rollout
        n_optimizations, # number of optimization steps
        embedding_space_size, # size of embeddings
        learning_rate, # learning rate of forward dynamics
        joint_training=False,
        using_extrinsic_reward=False
       ):
    # Initialize models
    emb = CnnEmbedding("embedding", env.observation_space, env.action_space, embedding_space_size)
    fd = ForwardDynamics("forward_dynamics", embedding_space_size, env.action_space) if not using_extrinsic_reward else None
    idf = InverseDynamics("inverse_dynamics", env.observation_space, env.action_space, embedding_space_size, emb) if idf else None
    policy = Policy("policy_new", env.action_space, joint_training, emb_size=embedding_space_size, emb_network=emb)
    ppo = PPO(env, policy,
              emb_network=emb, emb_size=embedding_space_size,
              max_timesteps=int(n_timesteps),
              clip_param=0.2, entcoeff=0.001,
              optim_epochs=8, optim_stepsize=1e-3, optim_batchsize=64,
              gamma=0.99, lam=0.95,
              schedule='linear',
              joint_training=joint_training,
             )

    if tensorboard:
        merged_summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter('tmp/tensorflow/', sess.graph)

    if not debug and rank == 0:
        cur_time = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        directory = 'results/' + cur_time
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory_m = 'model/' + cur_time
        if not os.path.exists(directory_m):
            os.makedirs(directory_m)

        txt = 'Running with env:%s, seed:%s, num timesteps:%s, joint-training:%s, using-extrinsic-reward:%s\n\n' \
               % (env_name, seed, n_timesteps, joint_training, using_extrinsic_reward)
        txt += 'Hyperparameters:\n - replay size:%s\n - batch size:%s\n - length of rollout:%s\n - number of optimization steps:%s\n - ' \
               % (replay_size, batch_size, len_rollouts, n_optimizations)
        txt += 'size of embedding:%s\n - learning rate of forward dynamics:%s\n\n' \
               % (embedding_space_size, learning_rate)
        txt += 'For inference on model, run:\n'
        txt += 'python3 cbf.py --env %s --seed %s --joint-training %s --using-extrinsic-reward %s ' \
               % (env_name, seed, joint_training, using_extrinsic_reward)
        txt += '--inference True --path-to-model %s' % (directory_m + '/model.ckpt')
        with open(directory + '/info.txt', 'w+') as txt_file:
            txt_file.write(txt)

    replay_memory = ReplayBuffer(replay_size)
    sess = tf.get_default_session()
    # sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    t = 0
    i = 0

    # initialize optimization batch variables
    a = env.action_space.sample() # not used, just so we have the datatype
    done = True # marks if we're on first timestep of an episode
    s = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    if joint_training:
        s_arr = np.array([np.zeros([84,84,4]) for _ in range(len_rollouts)])
    else:
        s_arr = np.array([np.zeros(embedding_space_size) for _ in range(len_rollouts)])
    r_arr = np.zeros(len_rollouts, 'float32')
    vpreds = np.zeros(len_rollouts, 'float32')
    dones = np.zeros(len_rollouts, 'int32')
    a_arr = np.array([a for _ in range(len_rollouts)])

    # For graphing
    best_reward = -float("inf")
    cur_reward = 0
    graph_rewards = []
    graph_best_rewards = []
    graph_epi_lens = []
    graph_in_rewards = []
    graph_avg_rewards = []
    graph_avg_epi_lens = []
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths

    while True:
        for j in range(len_rollouts):
            if not debug and rank == 0 and t > 0 and t % int(1e3) == 0:
                # print('# frame: %i. Best reward so far: %i.' % (t, best_reward,))
                save_to_file(directory, env_name, graph_rewards, graph_best_rewards, graph_epi_lens, graph_in_rewards, graph_avg_rewards, graph_avg_epi_lens)

                save_path = saver.save(sess, directory_m + '/model.ckpt')
                save_path = saver.save(sess, 'model/model.ckpt')
                #print("Model saved in file: %s" % save_path)

            if tensorboard and rank == 0 and t > 0 and t % int(1e3) == 0:
                summary = sess.run(merged_summary_op)
                writer.add_summary(summary, i)

            s = np.array(s)
            obs1 = emb.embed([s])
            if joint_training:
                a, vpred = policy.act([s])
            else:
                a, vpred = policy.act(obs1)

            # update optimization batch variables
            idx = t % len_rollouts
            s_arr[idx] = s if joint_training else obs1
            vpreds[idx] = vpred
            dones[idx] = done
            a_arr[idx] = a

            s_ , ext_r, done, _ = env.step(a)

            cur_reward += ext_r

            s_ = np.array(s_)

            # compute intrinsic reward
            obs2 = emb.embed([s_])
            r = fd.get_loss(obs1, obs2, np.eye(env.action_space.n)[a]) if not using_extrinsic_reward else ext_r
            
            replay_memory.add(s, a, r, s_, done)
            if t > 0 and t % int(2e2) == 0:
                graph_in_rewards.append((r, i))

            # update optimization batch variables
            r_arr[idx] = r
            cur_ep_ret += r
            cur_ep_len += 1

            # Prepare for next step
            if done:
                rewbuffer.append(cur_reward)
                lenbuffer.append(cur_ep_len)
                graph_rewards.append((cur_reward,i))
                graph_epi_lens.append((cur_ep_len,i))
                ep_rets.append(cur_reward)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                if cur_reward > best_reward:
                    best_reward = cur_reward
                graph_best_rewards.append((best_reward, i))
                graph_avg_rewards.append((sum(rewbuffer) / len(rewbuffer), i))
                graph_avg_epi_lens.append((sum(lenbuffer) / len(lenbuffer), i))
                cur_reward = 0
                s = env.reset()
            else:
                s = s_
            t += 1
            i += 1
        ppo.prepare({"ob" : s_arr, "rew" : r_arr, "vpred" : vpreds, "new" : dones,
                     "ac" : a_arr, "nextvpred": vpred * (1 - done),
                     "ep_rets" : ep_rets, "ep_lens" : ep_lens})
        ep_rets = []
        ep_lens = []
        for j in range(n_optimizations):
            # optimize theta_pi (and optionally theta_phi) wrt PPO loss
            ppo.step()
            # sample minibatch M from replay buffer R
            states, actions, rewards, next_states, _ = replay_memory.sample(batch_size)
            obs1, obs2 = emb.embed(states), emb.embed(next_states) # embedding of states
            actions_hot = np.squeeze([np.eye(env.action_space.n)[action] for action in actions])
            # optimize theta_f wtf forward dynamics loss on minibatch
            if not using_extrinsic_reward: fd.train(obs1, obs2, actions_hot, learning_rate)
            # optionally optimize theta_phi, theta_g wrt to auxilary loss
            if idf: idf.train(states, next_states, actions_hot, learning_rate)
        ppo.log()
        if ppo.timesteps_so_far >= n_timesteps:
            break
        i = ppo.timesteps_so_far

    if not debug and rank == 0:
        save_to_file(directory, env_name, graph_rewards, graph_best_rewards, graph_epi_lens, graph_in_rewards, graph_avg_rewards, graph_avg_epi_lens)


def main():
    args = parser.parse_args()
    with tf.Session() as sess:
        # env = gym.make(args.env)
        # initializing atari environment
        env = make_atari(args.env)
        env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)

        rank = MPI.COMM_WORLD.Get_rank()
        workerseed = args.seed + 10000 * rank
        set_global_seeds(workerseed)
        env.seed(workerseed)

        if args.inference:
            inference(env, sess, args.env,
                      path_to_model=args.path_to_model,
                      embedding_space_size=288,
                      joint_training=args.joint_training,
                      using_extrinsic_reward=args.using_extrinsic_reward,
                     )
        else:
            if rank == 0:
                logger.configure()
            else:
                logger.configure(format_strs=[])
                
            cbf(rank, env, sess, args.env, args.seed, args.debug,
                args.tensorboard, args.idf,
                replay_size=1000,
                batch_size=128,
                n_timesteps=args.num_timesteps,
                len_rollouts=256,
                n_optimizations=4,
                embedding_space_size=288,
                learning_rate=1e-5,
                joint_training=args.joint_training,
                using_extrinsic_reward=args.using_extrinsic_reward,
               )


if __name__ == '__main__':
    main()
