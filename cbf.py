# TODO integrate PPO
import numpy as np
import gym
import tensorflow as tf
from embedding import *
from policy import *
from forward_dynamics import *

from atari_wrappers import wrap_deepmind
from replay_memory import ReplayMemory


def cbf(env,
        n_rollouts,
        len_rollouts,
        n_optimizations,
        embedding_space_size,
        learning_rate,
        is_backprop_to_embedding=False):
    # Init
    emb = CnnEmbedding("embedding", env.observation_space, env.action_space, embedding_space_size)
    fd = ForwardDynamics("forward_dynamics", embedding_space_size, env.action_space)
    policy = Policy("policy", env.action_space, is_backprop_to_embedding, emb=emb, emb_space=embedding_space_size)
    replay_memory = ReplayMemory(REPLAY_SIZE)
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

    s = env.reset()
    t = 0

    for i in range(n_rollouts):
        for j in range(len_rollouts):

            s = np.array(s)
            obs1 = emb.embed(s)

            probs, val = policy.act(state=s, obs=obs1)
            a = np.random.choice(env.action_space.n, p=probs)
            s_ , _, done, _ = env.step(a) # ignoring reward from env

            s_ = np.array(s_)
            obs2 = emb.embed(s_)

            # compute intrinsic reward
            r = fd.get_loss(obs1, obs2, np.eye(env.action_space.n)[a])
            replay_memory.add((s, a, r, s_))

            # Prepare for next step
            t += 1
            if done:
                s = env.reset()
            else:
                s = s_
        for j in range(N_OPTIMIZATIONS):
            # optimize theta_pi (and optionally theta_phi) wrt PPO loss
            states, actions, rewards, next_states = replay_memory.sample(BATCH_SIZE) # minibatch from replay memory
            # optimize theta_f wtf forward dynamics loss on minibatch
            #fd.train(obs1, obs2, np.eye(env.action_space.n)[a], 0.5)
            #policy.train()

            # optionally optimize theta_phi, theta_A wrt to auxilary loss


N_ROLLOUTS = 1
LEN_ROLLOUTS = 1
N_OPTIMIZATIONS = 1
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

