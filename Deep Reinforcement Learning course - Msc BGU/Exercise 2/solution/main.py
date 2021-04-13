# -*- coding: utf-8 -*-

from World import World
import numpy as np

if __name__ == "__main__":
    env = World()
    env.reset()
    done = False
    t = 0
    env.show()
    while not done:
        env.render()
        print("state=", env.observation[0])
        action = np.random.randint(1, env.nActions + 1)
        # print("action=",action)
        next_state, reward, done = env.step(action)  # take a random action
        #env.render()
        # print("next_state",next_state)
        # print("env.observation[0]",env.observation[0])
        # print("done",done)
        # self.observation = [next_state];
        env.close()
        t += 1
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        #input()

    Q, policy, mean_reward, values = env.sarsa(num_episodes=14000, learning_rate=0.01, gamma=0.9, decay=50,values_filename='SARSA_values_14K_0.01_50.png', policy_filename='SARSA_policy_14K_0.01_50.png',action_values_filename='SARSA_actionValues_14K_0.01_50.png')
    Q, policy, mean_reward, values = env.Qlearning(num_episodes=25000, learning_rate=0.01, gamma=0.9, decay=500, values_filename='Qlearning_values_20K_0.01_500.png', policy_filename='Qlearning_policy_20K_0.01_500.png',action_values_filename='Qlearning_actionValues_20K_0.01_500.png')
    env.find_optimal_params('sarsa',
                        params_dict={'alpha': [0.01, 0.05, 0.1, 0.5], 'decay': [10, 50, 100, 500, 1000, 1500]},
                        target_vals=np.array(
                            [0, 0.285, 0.076, 0.008, 0.747, 0.576, 0, -0.085, 0.928, 0.584, 0.188, 0.08, 0, 0, 0,
                             -0.085]),
                        optimal_policy = np.array([1,2,1,1,2,1,1,2,2,1,1,1,1,1,1,4]))