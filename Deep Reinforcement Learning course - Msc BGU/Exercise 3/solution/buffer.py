import random
from collections import namedtuple

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, *args):
        """Saves a transition."""
        # TODO
        transition = Transition(args[0],args[1],args[2],args[3],args[4])
        if self.__len__() >= self.capacity:
            self.memory = self.memory[1:]
        self.memory.append(transition)


    def sample(self, batch_size):
        # TODO
        sample = random.sample(self.memory,batch_size)
        return sample


    def __len__(self):
        return len(self.memory)
