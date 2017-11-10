import numpy as np
import gym
import tensorflow as tf

from atari_wrappers import wrap_deepmind
from replay_memory import ReplayMemory

N = 0 # number of rollouts
N_opt = 0 # number of optimization steps
K = 0 # length of rollout
replay_size = 1000 # size of replay buffer
batch_size = 128 # size of minibatch from replay buffer

def frf(env, policy):
    replay_memory = ReplayMemory(replay_size)
    t = 0
    state = env.reset()
    for i in range(N):
        for j in range(K):
            action = policy.act() # assuming we have a policy class
            next_state, _, done, _ = env.step(action) # ignoring reward from env
            reward = 0 # compute intrinsic reward
            replay_memory.add((state, action, reward, next_state, done))
            t += 1
            if done:
                state = env.reset()
        for j in range(N_opt):
            # optimize theta_pi (and optionally theta_phi) wrt PPO loss
            states, actions, rewards, next_states, dones = replay_memory.sample(batch_size) # minibatch from replay memory
            # optimize theta_f wtf forward dynamics loss on minibatch
            # optionally optimize theta_phi, theta_A wrt to auxilary loss


if __name__ == '__main__':
    env = gym.make('Pong-v0')
    policy = None
    frf(env, policy)
