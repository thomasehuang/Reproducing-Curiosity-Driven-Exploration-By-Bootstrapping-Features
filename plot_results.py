import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='pong or seaquest', default='pong')
parser.add_argument('--type', help='best_reward or ep_len', default='ep_len')
#parser.add_argument('--type', help='best_reward or ep_len', default='best_reward')

import os

dir_path = os.getcwd()
def plot_ep_len(env, filter_size=100):

    ep_lens = []
    timesteps = []
    with open(dir_path+'/final_results/'+env+'/ep_len/frf.txt', 'r') as frf_ep_len_file:
        for line in frf_ep_len_file:
            ep_len, timestep = line.split()
            ep_lens.append(int(ep_len))
            timesteps.append(int(timestep) * 4)

    ep_lens2 = []
    timesteps2 = []
    with open(dir_path+'/final_results/'+env+'/ep_len/random.txt', 'r') as random_ep_len_file:
        for line in random_ep_len_file:
            ep_len, timestep = line.split()
            ep_lens2.append(ep_len)
            timesteps2.append(int(timestep) * 4)

    ep_lens3 = []
    timesteps3 = []
    with open(dir_path+'/final_results/'+env+'/ep_len/cbf.txt', 'r') as cbf_ep_len_file:
        for line in cbf_ep_len_file:
            ep_len, timestep = line.split()
            ep_lens3.append(ep_len)
            timesteps3.append(int(timestep) * 4)

    print('Finished reading file. Now graphing...')

    nbr_to_avg = 100;

    data_frf = { 'Training frames': timesteps, 'FRF Episode Length': ep_lens}
    d_frf = pd.DataFrame(data_frf)
    d_frf['FRF'] = d_frf['FRF Episode Length'].rolling(window=nbr_to_avg).mean()

    data_rand = {'Training frames': timesteps2, 'Random Episode Length': ep_lens2}
    d_rand = pd.DataFrame(data_rand)
    d_rand['RAND'] = d_rand['Random Episode Length'].rolling(window=nbr_to_avg).mean()

    data_cbf = {'Training frames': timesteps3, 'CBF Episode Length': ep_lens3}
    d_cbf = pd.DataFrame(data_cbf)
    d_cbf['CBF'] = d_cbf['CBF Episode Length'].rolling(window=nbr_to_avg).mean()

    ax = d_cbf.plot(x='Training frames', y='CBF', style='b-')
    ax = d_frf.plot(ax=ax, x='Training frames', y='FRF', style='b-.')
    d_rand.plot(ax=ax, x='Training frames', y='RAND', style='k-.')
    plt.ylabel('Episode length')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()


def plot_best_rew(env):
    best_rews = []
    timesteps = []
    with open(dir_path+'/final_results/'+env+'/best_reward/frf.txt', 'r') as frf_rew_file:
        for line in frf_rew_file:
            best_rew, timestep = line.split()
            best_rews.append(best_rew)
            timesteps.append(int(timestep) * 4)

    best_rews2 = []
    timesteps2 = []
    with open(dir_path+'/final_results/'+env+'/best_reward/random.txt', 'r') as random_rew_file:
        for line in random_rew_file:
            best_rew, timestep = line.split()
            best_rews2.append(best_rew)
            timesteps2.append(int(timestep) * 4)

    best_rews3 = []
    timesteps3 = []
    with open(dir_path+'/final_results/'+env+'/best_reward/cbf.txt', 'r') as cbf_rew_file:
        for line in cbf_rew_file:
            best_rew, timestep = line.split()
            best_rews3.append(best_rew)
            timesteps3.append(int(timestep) * 4)
    
    print('Finished reading file. Now graphing...')

    plt.xlabel('Training frames')
    plt.ylabel('Best return')
    plt.plot(timesteps, best_rews, 'b--', timesteps2, best_rews2, 'k--', timesteps3, best_rews3, 'b-')
    cbf = mlines.Line2D([], [], color='blue', linestyle='-',
                        markersize=15, label='CBF')
    frf = mlines.Line2D([], [], color='blue', linestyle='-.',
                        markersize=15, label='FRF')
    random = mlines.Line2D([], [], color='black', linestyle='-.',
                           markersize=15, label='RAND')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend(handles=[cbf, frf, random])
    plt.show()


def main():
    args = parser.parse_args()
    if args.type == 'best_reward':
        plot_best_rew(args.env)
    elif args.type == 'ep_len':
        plot_ep_len(args.env)


if __name__ == '__main__':
    main()
