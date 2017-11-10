import numpy as np
import gym
import matplotlib.pyplot as plt

from atari_wrappers import wrap_deepmind

TRAINING_FRAMES = int(2e8)

def main():
    env = wrap_deepmind(gym.make('Pong-v0'), episode_life=False, clip_rewards=False, frame_stack=True)

    nA = env.action_space.n

    rewards = []
    best_reward = -21
    cur_reward = 0
    state = env.reset()
    for i_f in range(1, TRAINING_FRAMES+1):
        if i_f % int(1e5) == 0:
            print('# frame: %i. Best reward so far: %i.' % (i_f, best_reward,))

        action = np.random.choice(nA)
        next_state, reward, done, _ = env.step(action)
        cur_reward += reward
        rewards.append(best_reward)
        if done:
            if cur_reward > best_reward:
                best_reward = cur_reward
            cur_reward = 0
            state = env.reset()
        else:
            state = next_state
    
    reward_file = open('results/pong_random_rewards.txt', 'w')
    for reward in rewards:
        reward_file.write("%s\n" % reward)

    plt.xlabel('Training frames')
    plt.ylabel('Best return')
    plt.plot(range(1, len(rewards)+1), rewards, 'k--')
    plt.show()


if __name__ == '__main__':
    main()
