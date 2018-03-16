from agents.dqn import DQN
from buffer.replay_buffer import ReplayBuffer
import gym
import argparse
import tensorflow as tf
import os
import numpy as np
import sys


class DQNDemo():
    def __init__(self, agent, env, max_episode=100, render=False):
        self.agent = agent
        self.env = env
        self.step = 0
        self.max_episode = max_episode
        self.ddqn = False
        self.render = render
        self.out_policy_file = os.path.join(
            os.getcwd(), 'model', 'dqn', 'policy_File_')

    def run(self):
        total_steps = 0
        total_reward = []
        for epi in range(self.max_episode):
            observation = self.env.reset()
            ep_r = 0
            while True:
                if self.render:
                    self.env.render()
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

                if self.agent.buffer.size() > 100 and total_steps % 5 == 0:
                    # print "Start training, sample Num so far:{}, episode Num so far:{}".format(
                    #     self.agent.buffer.size(), epi)
                    s_batch, a_batch, r_batch, s2_batch, t_batch = self.agent.buffer.sample()

                    # compute target
                    Q_target = self.agent.compute_target_q(
                        s_batch, a_batch, r_batch, s2_batch, t_batch)
                    Q_target = Q_target[:, np.newaxis]
                    self.agent.train(s_batch, a_batch, Q_target)

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
                print "Average reward of dqn agent is {}".format(np.mean(np.array(total_reward)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='dqn algorithm for cartpole task')
    parser.add_argument(
        '-s',
        '--seed',
        help='set seed number for training',
        type=int,
        default=0)  # obsolete (defined in config)
    parser.add_argument("-i", "--iteration", type=int)
    parser.add_argument(
        "-e",
        "--episodes",
        help="set the number of episodes",
        type=int,
        default=200)
    parser.add_argument(
        "--step",
        help="set the number of steps in each episode if the task is episode based",
        type=int)
    parser.set_defaults(use_color=True)
    parser.add_argument("--test", action="store_true")

    # Parameters for DRL
    parser.add_argument(
        "-b",
        "--batch",
        help="training batch for off policy algorithms",
        default=32,
        type=int)  # batch
    parser.add_argument(
        "-c",
        "--capacity",
        help="buffer size",
        default=20000,
        type=int)  # capacity

    parser.add_argument("-g", "--gamma")  # gamma
    parser.add_argument("--lr")
    parser.add_argument("--h1")
    parser.add_argument("--h2")

    # Params for AC
    # parser.add_argument("--alr")
    # parser.add_argument("--clr")

    # # Param for eigenoc
    # parser.add_argument('--alpha', help='weigth for intrinsic reward and external reward')
    # parser.add_argument('--term', help='termination factor')

    args = parser.parse_args()

    # initialization rl elements
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    sess = tf.Session()

    # seed setting
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # info about cartpole task
    # print env.action_space
    # print env.observation_space
    # print env.observation_space.high # bound for state
    # print env.observation_space.low

    buffer = ReplayBuffer(args.capacity, args.batch, args.seed)

    agent = DQN(
        sess,
        env.observation_space.shape[0],
        env.action_space.n,
        buffer=buffer)

    example = DQNDemo(agent, env, max_episode=args.episodes)
    example.run()
