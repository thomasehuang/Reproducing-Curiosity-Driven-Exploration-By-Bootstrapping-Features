import os
import datetime
import numpy as np
import argparse
import gym

from atari_wrappers import wrap_deepmind, make_atari
from baselines.common import set_global_seeds

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='gym environment ID', default='PongNoFrameskip-v4')
parser.add_argument('--seed', help='seed for environment', type=int, default=0)
parser.add_argument('--num-timesteps', help='number of timesteps', type=int, default=int(1e5))

def save_to_file(directory, env_name, graph_rewards, graph_epi_lens, graph_avg_rewards):
    if len(graph_rewards) != 0:
        with open(directory + '/' + env_name + '_best_rewards.txt', 'a+') as reward_file:
            for graph_reward, timestep in graph_rewards:
                reward_file.write("%s %s\n" % (graph_reward, timestep))
            graph_rewards[:] = []
    if len(graph_epi_lens) != 0:
        with open(directory + '/' + env_name + '_ep_len.txt', 'a+') as ep_len_file:
            for graph_epi_len, timestep in graph_epi_lens:
                ep_len_file.write("%s %s\n" % (graph_epi_len, timestep))
            graph_epi_lens[:] = []
    if len(graph_avg_rewards) != 0:
        with open(directory + '/' + env_name + '_avg_rewards.txt', 'a+') as avg_reward_file:
            for graph_avg_reward, timestep in graph_avg_rewards:
                avg_reward_file.write("%s %s\n" % (graph_avg_reward, timestep))
            graph_avg_rewards[:] = []


def main():
    args = parser.parse_args()

    env = make_atari(args.env)
    env = wrap_deepmind(env, clip_rewards=False, frame_stack=True, scale=True)
    set_global_seeds(args.seed)
    env.seed(args.seed)

    nA = env.action_space.n

    cur_time = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    directory = 'results/' + cur_time + '_random'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory_m = 'model/' + cur_time + '_random'
    if not os.path.exists(directory_m):
        os.makedirs(directory_m)

    # For graphing
    best_reward = -float("inf")
    cur_reward = 0
    cur_ep_len = 0
    sum_rewards = 0
    num_episodes = 0
    graph_rewards = []
    graph_epi_lens = []
    graph_avg_rewards = []

    _ = env.reset()
    for t in range(args.num_timesteps):
        if t > 0 and t % int(1e3) == 0:
            print('# frame: %i. Best reward so far: %i.' % (t, best_reward,))
            save_to_file(directory, args.env, graph_rewards, graph_epi_lens, graph_avg_rewards)

        action = np.random.choice(nA)
        _, reward, done, _ = env.step(action)
        cur_reward += reward
        cur_ep_len += 1
        if done:
            graph_epi_lens.append((cur_ep_len,t))
            cur_ep_len = 0
            if cur_reward > best_reward:
                best_reward = cur_reward
            graph_rewards.append((best_reward, t))
            sum_rewards += cur_reward
            num_episodes += 1
            graph_avg_rewards.append((sum_rewards / num_episodes, t))
            cur_reward = 0
            _ = env.reset()

    save_to_file(directory, env_name, graph_rewards, graph_epi_lens, graph_avg_rewards)


if __name__ == '__main__':
    main()
