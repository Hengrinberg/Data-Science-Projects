import numpy as np


def value_iteration(world, R, P, discount_factor, theta):
    V = np.zeros((world.nStates, 1))  # initiate all values to zero
    policy = np.zeros((world.nStates, 1))
    state_action_values = np.zeros((world.nStates, 4))

    while True:
        delta = 0
        for state in range(world.nStates):
            v = np.copy(V[state])
            action_values = np.zeros((world.nActions, 1))
            for action in range(world.nActions):
                action_values[action] = np.dot(P[action][state], R) + discount_factor * np.dot(P[action][state], V)
            if state not in [0, 6, 12, 13, 14]:
                V[state] = np.amax(action_values)
            delta = np.maximum(delta, abs(v - V[state]))
        if delta < theta:
            break

    for action in range(world.nActions):
        state_action_values[:, [action]] = np.dot(P[action], R) + discount_factor * np.dot(P[action], V)
    policy = np.argmax(state_action_values, axis=1)
    policy = policy + 1
    return V, policy


def policy_evaluation(policy, S, A, P, R,
                      discount_factor,
                      theta):


    values = np.zeros((S, 1))
    while True:
        delta = 0
        for state in range(S):
            v = values[state].copy()
            temp_val = 0
            if state not in [0, 6, 12, 13, 14]:
                for action in range(A):
                    discounted_value = (discount_factor * policy[state, action]) * np.dot(P[action][state],values)
                    reward = policy[state,action] * np.dot(P[action][state],R)
                    temp_val += reward + discounted_value
                values[state] = temp_val
            delta = max(delta, abs(v - values[state]))

        if delta < theta:
            break

    return values


def policy_improvement(S, A, P, R, V,
                      discount_factor):

    q_values = np.zeros((S, A))
    new_policy = np.zeros((S, A))
    for state in range(S):
        for action in range(A):
            temp=0
            if state not in [0, 6, 12, 13, 14]:
                temp = np.dot(P[action][state], V)
            q_values[state, action] = np.dot(P[action][state], R) + discount_factor * temp

    for state in range(S):
          if state not in [0, 6, 12, 13, 14]:
               max_val = np.max(q_values[state, :])
               best_actions = q_values[state, :] == max_val
               new_policy[state, :] = best_actions / sum(best_actions)

    return new_policy


def policy_iteration(world, S, A, P, R,
                      discount_factor,
                      theta):

    policy = np.full((world.nStates, world.nActions), 1 / world.nActions)
    policy_stable = False

    counter = 1
    while True:
        print('iteration_' + str(counter))
        V = policy_evaluation(policy, S, A, P, R, discount_factor, theta)
        new_policy = policy_improvement(S, A, P, R, V, discount_factor)
        world.plot_value(V, 'state_values_2e_iteration_' + str(counter) + '.png')
        world.plot_policy(np.argmax(new_policy, axis=1) + 1, 'policy_values_2e_iteration_' + str(counter) + '.png')
        counter += 1

        if (new_policy == policy).all():
            policy_stable = True
        policy = new_policy.copy()

        if policy_stable:
            break
    policy = np.argmax(policy, axis=1) + 1
    return policy, V



