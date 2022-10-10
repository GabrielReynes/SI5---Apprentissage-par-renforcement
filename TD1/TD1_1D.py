from collections import defaultdict
from random import random, choice

GRID_LENGTH = 6
GRID = [-1, 0, 0, 0, 0, 1]

EPSILON = 0.2
DISCOUNT = 0.9
ALPHA = 0.8
DISPLACEMENT_COST = 0.1
START_POS = 2

NB_TRAINING_ROUNDS = 100
NB_TESTING_ROUNDS = 10

ACTIONS = {"RIGHT": 1, "LEFT": -1}


def get_reward(state):
    return GRID[state] - DISPLACEMENT_COST


def create_q_table():
    return defaultdict(int)


def get_q_value(q_table, state, action):
    return q_table[(state, action)]


def get_legal_actions(state):
    return [key for key, value in ACTIONS.items() if 0 <= state + value < GRID_LENGTH]


def get_next_state(state, action):
    return state + ACTIONS[action]


def create_agent(start_state):
    return start_state, create_q_table()


def update_value(q_table, state, action, reward, next_state, alpha, discount):
    q_table[(state, action)] += \
        alpha * (reward
                 + discount * get_max_value(q_table, next_state)
                 - q_table[(state, action)])


def get_max_value(q_table, state):
    return max(get_q_value(q_table, state, action) for action in get_legal_actions(state))


def get_next_action(q_table, state, epsilon):
    legal_actions = get_legal_actions(state)
    if random() < epsilon:
        return choice(legal_actions)
    return max(legal_actions, key=lambda action: get_q_value(q_table, state, action))


def final_state(state):
    return state == 0 or state == GRID_LENGTH - 1


def explore(agent):
    state, q_table = agent
    while not final_state(state):
        action = get_next_action(q_table, state, EPSILON)
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        update_value(q_table, state, action, reward, next_state, ALPHA, DISCOUNT)
        state = next_state
    return state


def exploit(agent):
    state, q_table = agent
    nb_turn = 0
    while not final_state(state):
        nb_turn += 1
        action = get_next_action(q_table, state, 0)
        state = get_next_state(state, action)
    return state, nb_turn


def main():
    agent = create_agent(START_POS)

    train_successes = 0
    # NB SUCCESS - TOTAL NB TURN
    test_stats = [0, 0]

    for i in range(NB_TRAINING_ROUNDS):
        final_train_state = explore(agent)
        if final_train_state == GRID_LENGTH - 1:
            train_successes += 1

    for i in range(NB_TESTING_ROUNDS):
        final_test_state, nb_turn = exploit(agent)
        if final_test_state == GRID_LENGTH - 1:
            test_stats[0] += 1
        test_stats[1] += nb_turn

    print(f"Train Success : {train_successes / NB_TRAINING_ROUNDS * 100:.2f}% \n"
          f"Test Success : {test_stats[0] / NB_TESTING_ROUNDS * 100:.2f}%\n"
          f"Mean Nb Turn : {test_stats[1] / NB_TESTING_ROUNDS:.0f}\n")


main()
