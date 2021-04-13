import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
from collections import namedtuple
from itertools import count
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import box

from buffer import ReplayBuffer
from model import Network
from copy import copy
import pandas as pd

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'not_done'))

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
plt.ion()

# look for a gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Parameters
network_params = {
    'state_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'hidden_dim': 64
}

training_params = {
    'batch_size': 1,
    'gamma': 0.95,
    'epsilon_start': 1.1,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.95,
    'target_update': 'soft',  # use 'soft' or 'hard' or 'none
    'tau': 0.01,  # relevant for soft update
    'target_update_period': 15,  # relevant for hard update
    'grad_clip': 0.1,
}

network_params = box.Box(network_params)
params = box.Box(training_params)

# Build neural networks
policy_net = Network(network_params, device).to(device)
## TODO: build the target network and set its weights to policy_net's wights (use state_dict from pytorch)
target_net = Network(network_params, device).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


optimizer = optim.Adam(policy_net.parameters())
buffer = ReplayBuffer(100000)
#buffer = ReplayBuffer(1)
epsilon = params.epsilon_start


# ============================================================================
# Plotting function
def plot_graphs(all_scores, all_losses, all_errors, axes):
    axes[0].plot(range(len(all_scores)), all_scores, color='blue')
    axes[0].set_title('Score over episodes')
    axes[1].plot(range(len(all_losses)), all_losses, color='blue')
    axes[1].set_title('Loss over episodes')
    axes[2].plot(range(len(all_errors)), all_errors, color='blue')
    axes[2].set_title('Mean Q error over episodes')


# Training functions
def select_action(s):
    '''
    This function gets a state and returns an action.
    The function uses an epsilon-greedy policy.
    :param s: the current state of the environment
    :return: a tensor of size [1,1] (use 'return torch.tensor([[action]], device=device, dtype=torch.long)')
    '''
    # TODO: implement action selection.
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(network_params['action_dim'])]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return torch.tensor([[policy_net(s).max(1)[1].view(1,1)]], device=device, dtype=torch.long)



def train_model():
    # Pros tips: 1. There is no need for any loop here!!!!! Use matrices!
    #            2. Use the pseudo-code.


    if len(buffer) < params.batch_size:
        print('not enough samples')
        return 0, 0

    # sample mini-batch
    transitions = buffer.sample(params.batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    not_done_batch = torch.tensor(batch.not_done, device=device).bool()

    # Compute curr_Q = Q(s, a) - the model computes Q(s), then we select the columns of the taken actions.
    # Pros tips: First pass all s_batch through the network
    #            and then choose the relevant action for each state using the method 'gather'
    # TODO: fill curr_Q
    curr_Q = policy_net(state_batch).gather(1, action_batch)


    # Compute expected_Q (target value) for all states.
    # Don't forget that for terminal states we don't add the value of the next state.
    # Pros tips: Calculate the values for all next states ( Q_(s', max_a(Q_(s')) )
    #            and then mask next state's value with 0, where not_done is False (i.e., done).
    # TODO: fill expected_Q
    next_state_vals = torch.zeros(params.batch_size)
    try:
        not_last_states = torch.cat([state_ for ind, state_ in enumerate(batch.next_state) if not_done_batch[ind].item() is not False])
        if params.target_update == 'none':
            next_state_vals[not_done_batch] = policy_net.forward(not_last_states).max(1)[0].detach()
        else:
            next_state_vals[not_done_batch] = target_net.forward(not_last_states).max(1)[0].detach()

    except:
        print('there are no non last states')

    expected_Q = reward_batch + (params.gamma * next_state_vals)
    expected_Q = expected_Q.view(params.batch_size, 1)
    # Compute Huber loss. Smoother than MSE
    loss = F.smooth_l1_loss(curr_Q, expected_Q)

    # Optimize the model
    loss.backward()
    # clip gradients to help convergence
    nn.utils.clip_grad_norm_(policy_net.parameters(), params.grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    estimation_diff = torch.mean(curr_Q - expected_Q).item()

    return loss.item(), estimation_diff

# ============================================================================
def cartpole_play(weights_file_name):

    FPS = 25
    visualize = 'True'
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env,'recording',force=True)
    net = Network(network_params, device).to(device)
    print('load best model ...')
    net.load_state_dict(torch.load(weights_file_name))
    print('make movie ...')
    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False)).float()
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if visualize:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()



# ============================================================================

# Training loop

# train for max_episodes
initial_policy_net = Network(network_params, device).to(device)
initial_target_net = Network(network_params, device).to(device)
initial_policy_net.load_state_dict(policy_net.state_dict())
initial_target_net.load_state_dict(policy_net.state_dict())

max_episodes = 200
max_score = 500
task_scores = []
num_experiments = 5
highest_task_score = 0
# performances plots
experiment_name = 'replay_buffer_with_target_soft_update'
for experiment in range(1,num_experiments+1):
    fig, axes = plt.subplots(3, 1)
    all_scores = []
    all_losses = []
    all_errors = []

    task_score = 0
    if experiment > 1:
        buffer.memory = [] # reset buffer
        epsilon = params.epsilon_start # reset_epsilon
        # reset neural networks
        policy_net.load_state_dict(initial_policy_net.state_dict())
        target_net.load_state_dict(initial_target_net.state_dict())

    for i_episode in range(max_episodes):
        epsilon = max(epsilon*params.epsilon_decay, params.epsilon_end)
        ep_loss = []
        ep_error = []
        # Initialize the environment and state
        state = torch.tensor([env.reset()], device=device).float()
        done = False
        score = 0
        for t in count():
            # Select and perform an action
            action = select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            score += reward

            next_state = torch.tensor([next_state], device=device).float()
            reward = torch.tensor([reward], device=device).float()

            # Store the transition in memory
            buffer.push(state, action, next_state, reward, not done)

            # Update state
            state = next_state

            # Perform one optimization step (on the policy network)
            loss, Q_estimation_error = train_model()

            # save results
            ep_loss.append(loss)
            ep_error.append(Q_estimation_error)

            # soft target update
            if params.target_update == 'soft':
                # TODO: Implement soft target update.
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(params.tau * policy_param.data + (1.0 - params.tau) * target_param.data)


            if done or t >= max_score:
                print("Episode: {} | Current target score {} | Score: {}".format(i_episode+1, task_score, score))
                break

        # plot results

        all_scores.append(score)
        all_losses.append(np.average(ep_loss))
        all_errors.append(np.average(ep_error))
        plot_graphs(all_scores, all_losses, all_errors, axes)
        fig.canvas.draw()

        # hard target update. Copying all weights and biases in DQN
        if params.target_update == 'hard':
            # TODO: Implement hard target update.
            # Copy the weights from policy_net to target_net after every x episodes
            if i_episode % params.target_update_period == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print('updated')



        # update task score
        if min(all_scores[-5:]) > task_score:
            task_score = min(all_scores[-5:])
            if min(all_scores[-5:]) > highest_task_score:
                highest_task_score = min(all_scores[-5:])
                # TODO: store weights
                if params.target_update in ['soft', 'hard']:
                    torch.save(target_net.state_dict(), experiment_name + '_best.dat')
                else:
                    torch.save(policy_net.state_dict(), experiment_name + '_best.dat')



    task_scores.append(task_score)
    plt.savefig(experiment_name + '_' + str(experiment))
    print('added task score: ', task_score)




print('------------------------------------------------------------------------------')
print('Final task score = ', task_score)
print('Average task score = ', np.sum(np.array(task_scores)/len(task_scores)))
print(task_scores)

plt.ioff()
plt.show()

#cartpole_play('replay_buffer_with_target_with_hard_update_best.dat')

