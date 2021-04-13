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