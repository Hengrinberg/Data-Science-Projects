import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
from numpy.random import choice
import pandas as pd
import random
import time
import math
import pickle
from time import time



class World:

    def __init__(self):


        self.nRows = 4
        self.nCols = 4
        self.stateHoles = [1, 7, 14, 15]
        self.stateGoal = [13]
        self.nStates = 16
        self.States = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.nActions = 4
        self.rewards = np.array([-1] + [-0.04] * 5 + [-1] + [-0.04] * 5 + [1, -1, -1] + [-0.04])
        self.stateInitial = [4]
        self.observation =[]




    def _plot_world(self):
        plt.figure()
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        coord = [[0, 0], [nCols, 0], [nCols, nRows], [0, nRows], [0, 0]]
        xs, ys = zip(*coord)
        plt.plot(xs, ys, "black")
        for i in stateHoles:
            (I, J) = np.unravel_index(i, shape=(nRows, nCols), order='F')
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.5")
            plt.plot(xs, ys, "black")
        for ind, i in enumerate([stateGoal]):
            (I, J) = np.unravel_index(i, shape=(nRows, nCols), order='F')
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.8")
            plt.plot(xs, ys, "black")
        plt.plot(xs, ys, "black")
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        plt.plot(X, Y, 'k-')
        plt.plot(X.transpose(), Y.transpose(), 'k-')
        plt.savefig('plot_world.png')

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def plot(self):
        """
        plot function
        :return: None
        """
        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols

        self._plot_world()
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.5, j - 0.5, str(states[k]), fontsize=26, horizontalalignment='center',
                         verticalalignment='center')
                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        #plt.show(block=False)
        plt.show()


    def plot_value(self, valueFunction,filename):

        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        fig = plt.plot(1)
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateGoal:
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(valueFunction[k], 3)), fontsize=12, horizontalalignment='center', verticalalignment='center')
                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.savefig(filename)
        plt.show()


    def plot_actionValues(self, Q,filename):

        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        fig = plt.plot(1)
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateGoal:
                    plt.text(i + 0.5, j - 0.2, str(self._truncate(Q[k,0], 3)), fontsize=6, horizontalalignment='center', verticalalignment='top')
                    plt.text(i + 0.75, j - 0.5, str(self._truncate(Q[k, 1], 3)), fontsize=6,horizontalalignment='center', verticalalignment='center')
                    plt.text(i + 0.5, j - 0.8, str(self._truncate(Q[k, 2], 3)), fontsize=6,horizontalalignment='center', verticalalignment='bottom')
                    plt.text(i + 0.25, j - 0.5, str(self._truncate(Q[k, 3], 3)), fontsize=6,horizontalalignment='center', verticalalignment='center')
                    plt.plot([i, i+1], [j-1, j], 'k-', lw=0.8, color='gray')
                    plt.plot([i, i+1], [j, j-1], 'k-', lw=0.8, color='gray')
                k += 1
        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.savefig(filename)
        plt.show()

    def plot_policy(self, policy,filename):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        policy = policy.reshape(nRows, nCols, order="F").reshape(-1, 1)
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        X1 = X[:-1, :-1]
        Y1 = Y[:-1, :-1]
        X2 = X1.reshape(-1, 1) + 0.5
        Y2 = np.flip(Y1.reshape(-1, 1)) + 0.5
        X2 = np.kron(np.ones((1, nActions)), X2)
        Y2 = np.kron(np.ones((1, nActions)), Y2)
        mat = np.cumsum(np.ones((nStates, nActions)), axis=1).astype("int64")
        if policy.shape[1] == 1:
            policy = (np.kron(np.ones((1, nActions)), policy) == mat)
        index_no_policy = stateHoles + stateGoal
        index_policy = [item - 1 for item in range(1, nStates + 1) if item not in index_no_policy]
        mask = policy.astype("int64") * mat
        mask = mask.reshape(nRows, nCols, nCols)
        X3 = X2.reshape(nRows, nCols, nActions)
        Y3 = Y2.reshape(nRows, nCols, nActions)
        alpha = np.pi - np.pi / 2 * mask
        self._plot_world()
        for ii in index_policy:
            ax = plt.gca()
            j = int(ii / nRows)
            i = (ii + 1 - j * nRows) % nCols - 1
            index = np.where(mask[i, j] > 0)[0]
            h = ax.quiver(X3[i, j, index], Y3[i, j, index], np.cos(alpha[i, j, index]), np.sin(alpha[i, j, index]), 0.3)
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.25, j - 0.25, str(states[k]), fontsize=6, horizontalalignment='right', verticalalignment='bottom')
                k += 1
        plt.axis("equal")
        plt.axis("off")
        plt.savefig(filename)
        plt.show()



    def get_nrows(self):

        return self.nRows

    def get_ncols(self):

        return self.nCols

    def get_stateHoles(self):

        return self.stateHoles

    def get_stateGoal(self):

        return self.stateGoal

    def get_nstates(self):

        return self.nStates

    def get_nactions(self):

        return self.nActions


    def get_transition_model(self, p=0.8):
        nstates = self.nStates
        nrows = self.nRows
        holes_index = self.stateHoles
        goal_index = self.stateGoal
        terminal_index = holes_index + goal_index
        #actions = ["1", "2", "3", "4"]
        actions = [1, 2, 3, 4]     #I changed str to int
        transition_models = {}
        for action in actions:
            transition_model = np.zeros((nstates, nstates))
            for i in range(1, nstates + 1):
                if i not in terminal_index:
                    if action == 1:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                    if action == 2:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if 0 < i % nrows and (i + 1) <= nstates:
                            transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                    if action == 3:
                        if i + nrows <= nstates:
                            transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if 0 < i % nrows and (i + 1):
                            transition_model[i - 1][i + 1 - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                    if action == 4:
                        if 0 < i - nrows <= nstates:
                            transition_model[i - 1][i - nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if 0 < i % nrows and (i + 1) <= nstates:
                            transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if (i - 1) % nrows > 0:
                            transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                elif i in terminal_index:
                    transition_model[i - 1][i - 1] = 1

            transition_models[action] = pd.DataFrame(transition_model, index=range(1, nstates + 1),
                                                     columns=range(1, nstates + 1))
        return transition_models

    def step(self, action):
        observation = self.observation
        state = observation[0]
        #print('step -> state '+ str(state))
        #print('step -> action ' + str(action))
        prob = {}
        done = False
        transition_models = self.get_transition_model(0.8)
        #print(transition_models.keys())
        #print(transition_models[action])
        #print('inside')
        #print(state)
        #print(action)

        prob = transition_models[action].loc[state, :]
        #print(transition_models[action].loc[state, :])
        s = choice(self.States, 1, p=prob)
        next_state = s[0]
        reward = self.rewards[next_state - 1]

        if next_state in self.stateGoal + self.stateHoles:
            done = True
        self.observation = [next_state]
        return next_state, reward, done

    def reset(self, *args):
    #def reset(self):
        if not args:
            observation = self.stateInitial
        else:
            observation = []
            while not (observation):
                observation = np.setdiff1d(choice(self.States), self.stateHoles + self.stateGoal)
        self.observation = observation
        #return observation

    def render(self):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        observation = self.observation #observation
        state = observation[0]

        #state = 3

        J = nRows - (state-1) % nRows -1
        I = int((state-1)/nCols)


        circle = plt.Circle((I+0.5,J+0.5), 0.28, color='black')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

        self.plot()

        #plt.ion()
        #plt.show()
        #plt.draw()
        #plt.pause(0.5)
        #plt.ion()
        #plt.show(block=False)
        #time.sleep(1)
        # nRows = self.nRows
        # nCols = self.nCols
        # stateHoles = self.stateHoles
        # stateGoal = self.stateGoal


        #print(state)

        #circle = plt.Circle((0.5, 0.5), 0.1, color='black')
        #fig, ax = plt.subplots()
        #ax.add_artist(circle)

        # k = 0
        # for i in range(nCols):
        #     for j in range(nRows, 0, -1):
        #         if k + 1 not in stateHoles + stateGoal:
        #             plt.text(i + 0.5, j - 0.5, str(self._truncate(valueFunction[k], 3)), fontsize=12,
        #                      horizontalalignment='center', verticalalignment='center')
        #         k += 1


    def close(self):
        plt.pause(0.5)
        plt.close()

    def show(self):
        plt.ion()
        plt.show()

    # Get an action
    def get_action(self,Q, state, epsilon):
        values = Q[state, :]
        max_value = max(values)
        actions = [a for a in range(len(values))]
        greedy_actions = [a for a in range(len(values)) if values[a] == max_value]

        # Explore or get greedy
        if (random.random() < epsilon):
            return random.choice(actions)

        else:
            return random.choice(greedy_actions)

    def get_action_off_policy(self,Q, state):
        """ choose greedly the next action"""
        values = Q[state, :]
        max_ind = np.argmax(values)
        return max_ind

    # Update Q matrix
    def update(self, Q, current_state, next_state, reward, current_action, next_action, alpha=0.4, gamma=0.95):
        Q[current_state, current_action] = Q[current_state, current_action] + alpha * (
                    (reward + gamma * Q[next_state, next_action]) - Q[current_state, current_action])

    # Exploration rate
    def get_epsilon(seld,t,decay):
        return float(1/ (float(t/decay) + 1))

    def sarsa(self,num_episodes=10000,learning_rate=0.01,gamma=0.9,decay=10,values_filename='', policy_filename='',action_values_filename=''):
        Q = np.zeros((self.nStates, self.nActions))
        start = time()
        total_reward_list = []
        for episode in range(num_episodes):
            total_reward = 0
            print('episode_number: ' + str(episode+1))
            epsilon = self.get_epsilon(episode, decay)
            state = self.stateInitial[0]
            action = self.get_action(Q, state, epsilon)
            end_states = self.stateHoles + self.stateGoal
            while True:
                next_state, reward, done = self.step(action+1)
                total_reward += reward
                next_action = self.get_action(Q, next_state-1, epsilon)
                self.update(Q, state-1, next_state-1, reward, action, next_action, learning_rate, gamma)
                state = next_state
                action = next_action
                if next_state in end_states:
                    break
            total_reward_list.append(total_reward)
            self.reset()

        end = time()
        print('runtime: ' + str(start - end))
        policy = np.argmax(Q,axis=1) + 1
        V = np.amax(Q,axis=1)
        self.plot_actionValues(Q, action_values_filename)
        self.plot_value(V,values_filename)
        print(policy)
        print(Q)
        # Close the environment
        self.render()
        self.plot_policy(policy,policy_filename)
        self.close()

        return Q, policy, np.mean(total_reward_list), V

    def Qlearning(self,num_episodes=100,learning_rate=0.05,gamma=0.9,decay=10,values_filename='', policy_filename='',action_values_filename=''):
        Q = np.zeros((self.nStates, self.nActions))
        start = time()
        total_reward_list = []
        for episode in range(num_episodes):
            total_reward = 0
            print('episode_number: ' + str(episode+1))
            epsilon = self.get_epsilon(episode, decay)
            state = self.stateInitial[0]
            action = self.get_action(Q, state, epsilon)
            end_states = self.stateHoles + self.stateGoal
            while True:
                next_state, reward, done = self.step(action+1)
                total_reward += reward
                next_action = self.get_action_off_policy(Q, next_state - 1)
                self.update(Q, state-1, next_state-1, reward, action, next_action, learning_rate, gamma)
                state = next_state
                action = self.get_action(Q, next_state-1, epsilon)
                if next_state in end_states:
                    break
            total_reward_list.append(total_reward)
            self.reset()
        end = time()
        print('runtime: ' + str(start - end))
        policy = np.argmax(Q,axis=1) + 1
        V = np.amax(Q,axis=1)
        self.plot_value(V,values_filename)
        print(policy)
        print(Q)
        # Close the environment
        self.render()
        self.plot_actionValues(Q, action_values_filename)
        self.plot_policy(policy,policy_filename)
        self.close()

        return Q, policy, np.mean(total_reward_list), V


    def find_optimal_params(self,model_name,
                            params_dict={'alpha':[0.01,0.05,0.1,0.5],'decay':[10,50,100,500,1000,1500]},
                            target_vals=np.array([0,0.285,0.076,0.008,0.747,0.576,0,-0.085,0.928,0.584,0.188,0.08,0,0,0,-0.085]),
                            optimal_policy = np.array([1,2,1,1,2,1,1,2,2,1,1,1,1,1,1,4])):
        results = pd.DataFrame()
        alphas_list = []
        decays_list = []
        values_list = []
        error_from_target = []
        mean_reward_list = []
        is_optimal_policy_list = []
        if model_name == 'sarsa':
            model = self.sarsa()
        if model_name == 'qlearning':
            model = self.Qlearning()

        alphas = params_dict['alpha']
        decays = params_dict['decay']
        num_options = len(list(alphas)) * len(list(decays))
        counter = 0
        for alpha in alphas:
            for decay_ in decays:
                counter +=1
                alphas_list.append(alpha)
                decays_list.append(decay_)
                values_filename = model_name + '_values_10K_' + str(alpha) + '_' + str(decay_) + '.png'
                policy_filename = model_name + '_policy_10K_' + str(alpha) + '_' + str(decay_) + '.png'
                action_values_filename = model_name + '_actionValues_10K_' + str(alpha) + '_' + str(decay_) + '.png'
                Q, policy, mean_reward, values = self.sarsa(num_episodes=10000, learning_rate=alpha, gamma=0.9,
                                                           decay=decay_, values_filename=values_filename,
                                                           policy_filename=policy_filename,
                                                           action_values_filename=action_values_filename)
                error = np.absolute(target_vals - values)
                error_from_target.append(np.sum(error))
                mean_reward_list.append(mean_reward)
                values_list.append(values)
                is_same_policy = policy == optimal_policy
                is_optimal_policy_list.append(is_same_policy.all())
                print('finished ' + str(counter) + ' out of ' + str(num_options))

        results['alpha'] = alphas_list
        results['decay'] = decays_list
        results['values'] = values_list
        results['total_error'] = error_from_target
        results['mean_reward'] = mean_reward_list
        results['is_optimal_policy'] = is_optimal_policy_list
        results = results.sort_values(by='total_error')
        results.to_csv(model_name + '_result_summary.csv')










