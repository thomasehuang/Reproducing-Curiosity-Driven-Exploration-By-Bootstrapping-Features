import matplotlib.pyplot as plt


def plot_ep_len():
    ep_lens = []
    timesteps = []
    with open('results/pong_frf_ep_len.txt', 'r') as ep_len_file:
        for line in ep_len_file:
            ep_len, timestep = line.split()
            ep_lens.append(ep_len)
            timesteps.append(timestep)
    
    print('Finished reading file. Now graphing...')

    plt.xlabel('Training frames')
    plt.ylabel('Episode length')
    plt.plot(timesteps, ep_lens, 'b--', [1e6], [12e4], 'g--')
    plt.show()


def plot_in_rew():
    ep_in_rews = []
    timesteps = []
    with open('results/pong_frf_in_rewards.txt', 'r') as ep_in_rew_file:
        for line in ep_in_rew_file:
            ep_in_rew, timestep = line.split()
            ep_in_rews.append(ep_in_rew)
            timesteps.append(timestep)
    
    print('Finished reading file. Now graphing...')

    plt.xlabel('Training frames')
    plt.ylabel('Episode length')
    plt.plot(timesteps, ep_in_rews, 'b--')
    plt.show()


def main():
    plot_in_rew()


if __name__ == '__main__':
    main()
