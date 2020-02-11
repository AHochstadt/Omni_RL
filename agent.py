from methods import huber_loss, getLikelyTickSize
from utils import getChiTimeNow

import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K

from collections import deque

from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

class Agent:
    """ Stock Trading Bot """
    def __init__(self, model_cfg, config_dir, strategy="t-dqn", reset_every=200, searchForPretrained=True,
                 action_size=3, cross_q_prem=0, flat_q_prem=0):
        self.cross_q_prem = cross_q_prem
        self.flat_q_prem = flat_q_prem

        self.model_cfg = model_cfg
        self.config_dir = config_dir
        self.model_dir = self.config_dir + 'Deep_Models/' + self.model_cfg.model_str + '/'
        if not os.path.exists(self.model_dir):
            print(self.model_dir, 'not found. Creating it.')
            os.mkdir(self.model_dir)
            os.mkdir(self.model_dir+'Saved_Models/')
            os.mkdir(self.model_dir+'Trade_Logs/')

        self.strategy = strategy
        self.state_size = self.model_cfg.input_size
        self.action_size = action_size
        self.trade_log = None
        self.reset()
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        self.position = 0

        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
#         self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        self.num_loss_updates = 0
        self.num_target_updates = 0


        self.episode = 0
        self.ts_seen = 1
        if searchForPretrained:
            print('searching for pretrained model')
            self.model = self.loadModelWeights()
        else:
            self.model = self.model_cfg.model

        self.model.compile(loss=self.loss, optimizer=self.optimizer)

        self.reset_every = np.nan
        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())


    def setQPremiums(self, dataObj=None, cross_q_prem=None, flat_q_prem=None):
        """Premiums """
        assert (dataObj is not None or (cross_q_prem is not None and flat_q_prem is not None))
        if cross_q_prem is not None and flat_q_prem is not None:
            self.cross_q_prem = cross_q_prem
            self.flat_q_prem = flat_q_prem
        else:
            likelyTickSize = getLikelyTickSize(dataObj.train_close_pricesDF)
            self.cross_q_prem = .2*likelyTickSize
            self.flat_q_prem = .1*likelyTickSize

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))
        self.ts_seen += 1

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # make a definite buy on the first iter


        action_probs = self.model.predict(state)
        action_probs = list(action_probs[0]) #unwrap the unneccessary keras-mandated surrounding array

        # add cross q premium
        action_probs[1] += self.cross_q_prem
        action_probs[2] += self.cross_q_prem

        # add flat q premium
        if self.position == -1:
            action_probs[1] += self.flat_q_prem
        if self.position == 0:
            action_probs[0] += self.flat_q_prem
        if self.position == -1:
            action_probs[2] += self.flat_q_prem

        return np.argmax(action_probs[0])

    def getQValues(self, dataObj, t):
        state = self.getState(dataObj, t)
        q_values = self.model.predict(state)
        return q_values

    def getState(self, dataObj, t):
        """Returns the position and the t-th row of the data"""
        ret_array = [self.position] + list(dataObj.trainDF.iloc[t].values)
        ret_array = np.reshape(ret_array, len(ret_array))
        return(np.array([ret_array]))

    def sample_experiences(self, batch_size):
        """
        Sample previous experiences in memory.
        Also updates target network for applicable strategies.
        """
        sample_dist = np.array([i for i in range(len(self.memory))])
        sample_dist = sample_dist/sum(sample_dist)

        mini_batch_idx = np.random.choice(range(len(self.memory)), batch_size, p=sample_dist)
        mini_batch = [self.memory[i] for i in mini_batch_idx]
        X_train, y_train = [], []

        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
#             if self.ts_seen % self.reset_every == 0:
            if self.ts_seen % self.reset_every < batch_size: # this is valid because now we're only training every batch_size timesteps
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())
                self.num_target_updates += 1

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation with fixed targets
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # Double DQN
        elif self.strategy == "double-dqn":
#             if self.ts_seen % self.reset_every == 0:
            if self.ts_seen % self.reset_every < batch_size: # this is valid because now we're only training every batch_size timesteps
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())
                self.num_target_updates += 1

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate double deep q-learning equation
                    target = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        else:
            raise NotImplementedError()
        return X_train, y_train


    def evaluate_experience_replay(self, batch_size):
        """For use in evaluation. Takes the loss of previous experiences in memory without fitting."""
        X_train, y_train = self.sample_experiences(batch_size)

        loss = self.model.evaluate(
            np.array(X_train), np.array(y_train),
            verbose=0)#.history["loss"][0]

        return loss

    def train_experience_replay(self, batch_size):
        """Sample from previous experiences and train model."""
        X_train, y_train = self.sample_experiences(batch_size)

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.num_loss_updates += 1

        return loss

    def saveTradeLog(self, sess_type='train'):
        # if there's already a file there, move it
        fn = self.model_dir+'Trade_Logs/' + str(self.episode) + '.csv'
        if sess_type != 'train':
            fn = self.model_dir+'Trade_Logs/' + sess_type + str(self.episode) + '.csv'
        if os.path.exists(fn):
            print(fn, 'already exists! Moving to Previous/')
            prev_dir = self.model_dir+'Trade_Logs/Previous/'
            if not os.path.exists(prev_dir):
                os.mkdir(prev_dir)
            os.replace(fn, prev_dir+str(self.episode) + str(getChiTimeNow())[:10] + '.csv')
        self.trade_log.to_csv(fn, index=False)

    def saveModelWeights(self):
        # if there's already a file there, move it
        fn = self.model_dir+'Saved_Models/'+str(self.episode)+'.h5'
        if os.path.exists(fn):
            print(fn, 'already exists! Moving to Previous/')
            prev_dir = self.model_dir+'Saved_Models/Previous/'
            if not os.path.exists(prev_dir):
                os.mkdir(prev_dir)
            os.replace(fn, prev_dir+str(self.episode) + str(getChiTimeNow())[:10] + '.h5')
        self.model.save_weights(fn)


    def loadModelWeights(self):
        saved_models_dir = self.model_dir + 'Saved_Models/'
        saved_models = os.listdir(saved_models_dir)
        model = self.model_cfg.model
        if len(saved_models) == 0:
            print('No models saved. Starting from episode 0 with a new model.')
        else:
            episodes_saved = [int(i.split('.')[0]) for i in saved_models if i.split('.')[0].isdigit()]
            print('Found models. Loading episode', max(episodes_saved))
            try:
                model.load_weights(saved_models_dir+str(max(episodes_saved))+'.h5')
            except:
                #if there is an error loading the latest model
                #   (possibly because of an interruption during saving) ... try the next-latest model
                model.load_weights(saved_models_dir+str(max(episodes_saved)-1)+'.h5')
            self.episode = max(episodes_saved) +1
        return model

    def reset(self):
        self.trade_log = pd.DataFrame(columns = ['Minute', 'DesiredAction', 'ActualAction', 'NewPosition', 'C_B', 'C_A'])
        self.position = 0
        self.num_loss_updates = 0
        self.num_target_updates = 0
        self.ts_seen = 1

    def updateTradeLog(self, minute, action, actual_action, new_position, close_prices):
        self.trade_log = self.trade_log.append({'Minute': minute,
                                                'DesiredAction': action,
                                                'ActualAction': actual_action,
                                                'NewPosition': new_position,
                                                'C_B': close_prices.C_B1,
                                                'C_A': close_prices.C_A1}, ignore_index=True)
    def getNumTrades(self):
        return(int((self.trade_log.ActualAction != 0).sum()))

    def viewResults(self):
        print('viewing results')
        # construct pnl
        full_logDF = self.trade_log.copy()
        full_logDF['Midpt'] = (full_logDF.C_B + full_logDF.C_A)/2
        full_logDF['DayPNL'] = (full_logDF.ActualAction!=0)*(full_logDF.C_B - full_logDF.Midpt) #it's always negative half the bid-ask spread
        full_logDF['OpenPNL'] = full_logDF.NewPosition.shift() * (full_logDF.Midpt - full_logDF.Midpt.shift())
        full_logDF.OpenPNL.loc[0] = 0
        full_logDF['TotalPNL'] = full_logDF.DayPNL + full_logDF.OpenPNL
        full_logDF['CumPNL'] = full_logDF.TotalPNL.cumsum()
        # [x] graph (from http://kitchingroup.cheme.cmu.edu/blog/2013/09/13/Plotting-two-datasets-with-very-different-scales/)
        buys = full_logDF.loc[full_logDF.ActualAction == 1][['Minute', 'C_A']]
        sells = full_logDF.loc[full_logDF.ActualAction == 2][['Minute', 'C_B']]
        print('viewing figure')

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(full_logDF.Minute, full_logDF.Midpt, 'k-', linewidth=0.5)
        ax1.plot(buys.Minute, buys.C_A, 'g+')
        ax1.plot(sells.Minute, sells.C_B, 'r+')
        ax1.set_ylabel('Price')

        ax2 = ax1.twinx()
        ax2.plot(full_logDF.Minute, full_logDF.CumPNL, 'b-')
        ax2.set_ylabel('PNL')

        plt.show()

        # [] check that the total PNL sum is close
