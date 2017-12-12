from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

from atari_wrappers import wrap_deepmind
import gym

class PPO(object):
    def __init__(self, env, policy_new, policy_old,
                 timesteps_per_actorbatch, # timesteps per actor per update
                 clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
                 optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
                 gamma, lam, # advantage estimation
                 max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
                 adam_epsilon=1e-5,
                 schedule='constant',
                 joint_training=False
                 ):
        # Setup variables
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.gamma = gamma
        self.lam = lam
        self.max_timesteps = max_timesteps
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule

        # Setup losses and stuff
        # ----------------------------------------
        ob_space = env.observation_space
        ac_space = env.action_space
        self.pi = policy_new # Construct network for new policy
        oldpi = policy_old # Network for old policy
        atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
        clip_param = clip_param * lrmult # Annealed cliping parameter epislon

        # ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        if joint_training:
            ob = U.get_placeholder_cached(name="ob_f")
        else:
            ob = U.get_placeholder_cached(name="ob")
        ac = self.pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(self.pi.pd)
        ent = self.pi.pd.entropy()
        meankl = U.mean(kloldnew)
        meanent = U.mean(ent)
        pol_entpen = (-entcoeff) * meanent

        ratio = tf.exp(self.pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
        surr1 = ratio * atarg # surrogate from conservative policy iteration
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
        pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = U.mean(tf.square(self.pi.vpred - ret))
        self.total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        var_list = self.pi.get_trainable_variables()
        self.lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(self.total_loss, var_list)])
        self.adam = MpiAdam(var_list, epsilon=adam_epsilon)

        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpi.get_variables(), self.pi.get_variables())])
        self.compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

        U.initialize()
        self.adam.sync()

        # Prepare for rollouts
        # ----------------------------------------
        self.episodes_so_far = 0
        self.timesteps_so_far = 0
        self.iters_so_far = 0
        self.tstart = time.time()
        self.lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

        # self.train_step = tf.train.AdamOptimizer(adam_epsilon).minimize(self.total_loss, var_list=var_list)
        # self.train = U.function([ob, ac, atarg, ret, lrmult], self.train_step)


    def prepare(self, batch):
        if self.schedule == 'constant':
            self.cur_lrmult = 1.0
        elif self.schedule == 'linear':
            self.cur_lrmult =  max(1.0 - float(self.timesteps_so_far) / self.max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%self.iters_so_far)

        # seg = seg_gen.__next__() # generate next sequence
        self.seg = batch
        self.add_vtarg_and_adv(self.seg, self.gamma, self.lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, self.tdlamret = self.seg["ob"], self.seg["ac"], self.seg["adv"], self.seg["tdlamret"]
        self.vpredbefore = self.seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        self.d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=self.tdlamret), shuffle=not self.pi.recurrent)
        self.optim_batchsize = self.optim_batchsize or ob.shape[0]

        self.assign_old_eq_new() # set old parameter values to new parameter values


    def step(self):
        # Here we do a bunch of optimization epochs over the data
        for _ in range(self.optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for b in self.d.iterate_once(self.optim_batchsize):
                # self.train(b["ob"], b["ac"], b["atarg"], b["vtarg"], self.cur_lrmult)
                *newlosses, g = self.lossandgrad(b["ob"], b["ac"], b["atarg"], b["vtarg"], self.cur_lrmult)
                self.adam.update(g, self.optim_stepsize * self.cur_lrmult)
                # losses.append(newlosses)
            # logger.log(fmt_row(13, np.mean(losses, axis=0)))

    
    def log(self):
        logger.log("Evaluating losses...")
        losses = []
        for b in self.d.iterate_once(self.optim_batchsize):
            newlosses = self.compute_losses(b["ob"], b["ac"], b["atarg"], b["vtarg"], self.cur_lrmult)
            losses.append(newlosses)            
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, self.loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(self.vpredbefore, self.tdlamret))
        lrlocal = (self.seg["ep_lens"], self.seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(self.flatten_lists, zip(*listoflrpairs))
        self.lenbuffer.extend(lens)
        self.rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(self.lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(self.rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        self.episodes_so_far += len(lens)
        self.timesteps_so_far += sum(lens)
        self.iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", self.episodes_so_far)
        logger.record_tabular("TimestepsSoFar", self.timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - self.tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

    def add_vtarg_and_adv(self, seg, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]

    def flatten_lists(self, listoflists):
        return [el for list_ in listoflists for el in list_]
