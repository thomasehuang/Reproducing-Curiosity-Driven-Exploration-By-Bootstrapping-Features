import numpy as np
import gym
import tensorflow as tf
from embedding import *
from policy import *

from atari_wrappers import wrap_deepmind
from replay_memory import ReplayMemory


N_ROLLOUTS = 10
LEN_ROLLOUTS = 10
N_OPTIMIZATIONS = 1
REPLAY_SIZE = 1000
BATCH_SIZE = 128


def cbf(env, policy):
    replay_memory = ReplayMemory(REPLAY_SIZE)
    t = 0
    s = env.reset()
    for i in range(N_ROLLOUTS):
        for j in range(LEN_ROLLOUTS):
            #a = policy.act() # assuming we have a policy class
            a = env.action_space.sample()
            s_ , _, done, _ = env.step(a) # ignoring reward from env
            r = 0 # compute intrinsic reward TODO implement f and phi
            replay_memory.add((s, a, r, s_))

            t += 1
            if done:
                s = env.reset()
            else:
                s = s_
        for j in range(N_OPTIMIZATIONS):
            # optimize theta_pi (and optionally theta_phi) wrt PPO loss
            states, actions, rewards, next_states = replay_memory.sample(BATCH_SIZE) # minibatch from replay memory
            # optimize theta_f wtf forward dynamics loss on minibatch
            # optionally optimize theta_phi, theta_A wrt to auxilary loss


if __name__ == '__main__':
    env = wrap_deepmind(gym.make('Pong-v0'), episode_life=False, clip_rewards=False, frame_stack=True)
    policy = None
    #cbf(env, policy)

    s = env.reset()
    s_arr = np.array(s)

    with tf.Session() as sess:

        emb = CnnEmbedding("embedding", env.observation_space, env.action_space)
        policy = Policy("policy", env.action_space)

        sess.run(tf.global_variables_initializer())

        obs = emb.embed(s_arr)
        print(tf.shape(obs))
        probs, value = policy.act(obs[0])
        print(probs, value)


