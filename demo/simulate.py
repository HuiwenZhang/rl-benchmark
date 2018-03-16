from agents.pg import PolicyGradient
from buffer.replay_buffer_episode import ReplayBufferEpisode
import gym
import argparse
import tensorflow as tf
import os
import numpy as np
import sys


class SimulateDemo(object):
    def __init__(self, agent, env, max_episode=100, render=False):
        self.agent = agent
        self.agent_name = 'general agent'
        self.env = env
        self.step = 0
        self.max_episode = max_episode
        self.render = render
        self.out_policy_file = os.path.join(
            os.getcwd(), 'model', self.agent_name, 'policy_File_')

    def process_sample(self):
        """
        In a simulation process, we do three things: 1) take action; 2) training; 3)record results
        Different agents use different buffer and training with different input. This func will be overide by child method
        :return:
        """
        return self.agent.buffer.sample()

    def run(self):
        total_steps = 0
        total_reward = []
        for epi in range(self.max_episode):
            observation = self.env.reset()
            ep_r = 0
            while True:
                if self.render:
                    self.env.render()
                # choose aciton
                action = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)

                # the smaller theta and closer to center the better
                # x, x_dot, theta, theta_dot = observation_
                # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                # reward = r1 + r2

                # store experience
                self.agent.buffer.record(
                    observation, action, reward, observation_)
                ep_r += reward

                if self.agent.buffer.size() > 2 and total_steps % 5 == 0:
                    # print "Start training, sample Num so far:{}, episode Num so far:{}".format(
                    #     self.agent.buffer.size(), epi)
                    s_batch, a_batch, r_batch, s2_batch, t_batch = self.agent.buffer.sample()

                    # compute cumulative reward
                    epi_reward = []
                    for r in r_batch:
                        dis_reward = self.agent.discount_reward(r)
                        epi_reward += dis_reward

                    self.agent.train(
                        np.concatenate(
                            np.array(s_batch),
                            axis=0).tolist(),
                        np.concatenate(
                            np.array(a_batch),
                            axis=0).tolist(),
                        epi_reward)

                if done:
                    total_reward.append(ep_r)
                    print 'Episode: {}'.format(
                        epi) + '\t' + 'episode reward: {}'.format(ep_r)
                    break
                observation = observation_
                total_steps += 1

            # Save model
            if epi == (self.max_episode - 1):
                pol_file_name = self.out_policy_file + str(epi)
                self.agent.savePolicy(pol_file_name)
                print "Average reward of {} agent is {}".format(self.agent_name, np.mean(np.array(total_reward)))



