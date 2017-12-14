import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='pong or seaquest', default='pong')
parser.add_argument('--type', help='best_reward or ep_len', default='best_reward')


def plot_ep_len(env):
    ep_lens = []
    timesteps = []
    with open('final_results/'+env+'/ep_len/frf.txt', 'r') as frf_ep_len_file:
        for line in frf_ep_len_file:
            ep_len, timestep = line.split()
            ep_lens.append(int(ep_len))
            timesteps.append(int(timestep) * 4)

    ep_lens2 = []
    timesteps2 = []
    with open('final_results/'+env+'/ep_len/random.txt', 'r') as random_ep_len_file:
        for line in random_ep_len_file:
            ep_len, timestep = line.split()
            ep_lens2.append(ep_len)
            timesteps2.append(int(timestep) * 4)

    ep_lens3 = []
    timesteps3 = []
    with open('final_results/'+env+'/ep_len/cbf.txt', 'r') as cbf_ep_len_file:
        for line in cbf_ep_len_file:
            ep_len, timestep = line.split()
            ep_lens3.append(ep_len)
            timesteps3.append(int(timestep) * 4)
    
    print('Finished reading file. Now graphing...')

    plt.xlabel('Training frames')
    plt.ylabel('Episode length')
    plt.plot(timesteps, ep_lens, 'b--', timesteps2, ep_lens2, 'k--', timesteps3, ep_lens3, 'b-')
    cbf = mlines.Line2D([], [], color='blue', linestyle='-',
                        markersize=15, label='CBF')
    frf = mlines.Line2D([], [], color='blue', linestyle='--',
                        markersize=15, label='FRF')
    random = mlines.Line2D([], [], color='black', linestyle='--',
                           markersize=15, label='RAND')
    plt.legend(handles=[cbf, frf, random])
    plt.show()


def plot_best_rew(env):
    best_rews = []
    timesteps = []
    with open('final_results/best_reward/'+env+'/frf.txt', 'r') as frf_rew_file:
        for line in frf_rew_file:
            best_rew, timestep = line.split()
            best_rews.append(best_rew)
            timesteps.append(int(timestep) * 4)

    best_rews2 = []
    timesteps2 = []
    with open('final_results/best_reward/'+env+'/random.txt', 'r') as random_rew_file:
        for line in random_rew_file:
            best_rew, timestep = line.split()
            best_rews2.append(best_rew)
            timesteps2.append(int(timestep) * 4)

    best_rews3 = []
    timesteps3 = []
    with open('final_results/best_reward/'+env+'/cbf.txt', 'r') as cbf_rew_file:
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
    frf = mlines.Line2D([], [], color='blue', linestyle='--',
                        markersize=15, label='FRF')
    random = mlines.Line2D([], [], color='black', linestyle='--',
                           markersize=15, label='RAND')
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
