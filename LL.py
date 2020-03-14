import gym
import numpy as np
import pandas as pd
from collections import deque
import random

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model


import pickle
from matplotlib import pyplot as plt

import time
millis = int(round(time.time()))

class DQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rewards_list = []

        self.replay_memory = deque(maxlen=500000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.action_num = self.action_space.n
        self.observation_num = env.observation_space.shape[0]
        self.model = self.initialize_model()

    def initialize_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.observation_num, activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(self.action_num, activation=linear))

        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_num)

        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def reply(self):

        if len(self.replay_memory) < self.batch_size or self.counter != 0:
            return

        if np.mean(self.rewards_list[-10:]) > 180:
            return

        random_sample = self.randsample()
        states, actions, rewards, next_states, done_list = self.sampleatt(random_sample)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)


    def randsample(self):
        random_sample = random.sample(self.replay_memory, self.batch_size)
        return random_sample
    def sampleatt(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list

    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size

    def save(self, name):
        self.model.save(name)

    def train(self, num_episodes=2000, can_stop=True):
        print("Training model...")
        for episode in range(num_episodes):
            state = env.reset()
            reward_for_episode = 0
            num_steps = 1000
            state = np.reshape(state, [1, self.observation_num])
            for step in range(num_steps):
                env.render()
                received_action = self.get_action(state)
                next_state, reward, done, info = env.step(received_action)
                next_state = np.reshape(next_state, [1, self.observation_num])
                self.add_to_replay_memory(state, received_action, reward, next_state, done)
                reward_for_episode += reward
                state = next_state
                self.update_counter()
                self.reply()

                if done:
                    break
            self.rewards_list.append(reward_for_episode)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if last_rewards_mean > 200 and can_stop:
                break
            print("Episode: ", episode, "\tReward: ",reward_for_episode, "\tAverage Reward: ",last_rewards_mean)


def testmodel(trained_model):
    rewards_list = []
    num_test_episode = 100
    env = gym.make("LunarLander-v2")
    print("Testing of the trained model...")

    step_count = 1000

    for test_episode in range(num_test_episode):
        current_state = env.reset()
        observation_num = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, observation_num])
        reward_for_episode = 0
        for step in range(step_count):
            env.render()
            selected_action = np.argmax(trained_model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, observation_num])
            current_state = new_state
            reward_for_episode += reward
            if done:
                break
        rewards_list.append(reward_for_episode)
        print("Episode: ", test_episode, "\tReward: ", reward_for_episode)

    return rewards_list


def gammaexperiment():
    print('Running Experiment for gamma...')
    env = gym.make('LunarLander-v2')

    env.seed(millis)
    np.random.seed(millis)

    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma_list = [0.99, 0.8, 0.6, 0.5]
    training_episodes = 1000

    rewards_list_for_gammas = []
    for gamma_value in gamma_list:
        model = DQN(env, lr, gamma_value, epsilon, epsilon_decay)
        print("Training model for Gamma: {}".format(gamma_value))
        model.train(training_episodes, False)
        rewards_list_for_gammas.append(model.rewards_list)

    pickle.dump(rewards_list_for_gammas, open("rewards_list_for_gammas.p", "wb"))
    rewards_list_for_gammas = pickle.load(open("rewards_list_for_gammas.p", "rb"))

    gamma_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(gamma_list)):
        col_name = "gamma=" + str(gamma_list[i])
        gamma_rewards_pd[col_name] = rewards_list_for_gammas[i]
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = gamma_rewards_pd.plot(linewidth=1, figsize=(15, 8), title="Rewards per episode for different gamma values")
    plot.set_xlabel("Episodes")
    plot.set_ylabel("Reward")
    plt.ylim((-600, 300))
    fig = plot.get_figure()
    fig.savefig("Rewards per episode for different gamma values")


def lrexperiment():
    print('Running Experiment for learning rate...')
    env = gym.make('LunarLander-v2')

    env.seed(millis)
    np.random.seed(millis)

    lr_values = [0.0001, 0.001, 0.01, 0.1]
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 1000
    rewards_list_for_lrs = []
    for lr_value in lr_values:
        model = DQN(env, lr_value, gamma, epsilon, epsilon_decay)
        print("Training model for LR: {}".format(lr_value))
        model.train(training_episodes, False)
        rewards_list_for_lrs.append(model.rewards_list)

    pickle.dump(rewards_list_for_lrs, open("rewards_list_for_lrs.p", "wb"))
    rewards_list_for_lrs = pickle.load(open("rewards_list_for_lrs.p", "rb"))

    lr_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(lr_values)):
        col_name = "lr="+ str(lr_values[i])
        lr_rewards_pd[col_name] = rewards_list_for_lrs[i]
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = lr_rewards_pd.plot(linewidth=1, figsize=(15, 8), title="Rewards per episode for different learning rates")
    plot.set_xlabel("Episodes")
    plot.set_ylabel("Reward")
    plt.ylim((-2000, 300))
    fig = plot.get_figure()
    fig.savefig("Rewards per episode for different learning rates")

def edexperiment():
    print('Running Experiment for epsilon decay...')
    env = gym.make('LunarLander-v2')

    env.seed(millis)
    np.random.seed(millis)

    lr = 0.001
    epsilon = 1.0
    ed_values = [0.999, 0.995, 0.990, 0.950]
    gamma = 0.99
    training_episodes = 1000

    rewards_list_for_ed = []
    for ed in ed_values:
        model = DQN(env, lr, gamma, epsilon, ed)
        print("Training model for ED: {}".format(ed))
        model.train(training_episodes, False)
        rewards_list_for_ed.append(model.rewards_list)

    pickle.dump(rewards_list_for_ed, open("rewards_list_for_ed.p", "wb"))
    rewards_list_for_ed = pickle.load(open("rewards_list_for_ed.p", "rb"))

    ed_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes+1)))
    for i in range(len(ed_values)):
        col_name = "epsilon_decay = "+ str(ed_values[i])
        ed_rewards_pd[col_name] = rewards_list_for_ed[i]
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = ed_rewards_pd.plot(linewidth=1, figsize=(15, 8), title="Rewards per episode for different epsilon(ε) decay")
    plot.set_xlabel("Episodes")
    plot.set_ylabel("Reward")
    plt.ylim((-600, 300))
    fig = plot.get_figure()
    fig.savefig("Rewards per episode for different epsilon(ε) decay")

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    env.seed(millis)
    np.random.seed(millis)

    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 2000
    model = DQN(env, lr, gamma, epsilon, epsilon_decay)
    model.train(training_episodes, True)

    save_dir = "saved_models"
    model.save(save_dir + "trained_model.h5")
    pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list.p", "wb"))
    rewards_list = pickle.load(open(save_dir + "train_rewards_list.p", "rb"))

    reward_df = pd.DataFrame(rewards_list)
    plt.rcParams.update({'font.size': 17})
    reward_df['rolling_mean'] = reward_df[reward_df.columns[0]].rolling(100).mean()
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = reward_df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel('Episode')
    plot.set_ylabel('Reward')
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig("Reward for each training episode")

    trained_model = load_model(save_dir + "trained_model.h5")
    test_rewards = testmodel(trained_model)
    pickle.dump(test_rewards, open(save_dir + "test_rewards.p", "wb"))
    test_rewards = pickle.load(open(save_dir + "test_rewards.p", "rb"))

    testreward_df = pd.DataFrame(test_rewards)
    testreward_df['mean'] = testreward_df[testreward_df.columns[0]].mean()
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = testreward_df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel('Episode')
    plot.set_ylabel('Reward')
    plt.ylim((0, 300))
    plt.xlim((0, 100))
    plt.legend().set_visible(False)
    fig = plot.get_figure()
    fig.savefig("Reward for each testing episode")
    print("Training and Testing Completed!")

    lrexperiment()
    edexperiment()
    gammaexperiment()
