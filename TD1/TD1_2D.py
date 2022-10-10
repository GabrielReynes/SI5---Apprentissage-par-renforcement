from collections import defaultdict
from random import random, choice

# Classic toy example
GRID_WIDTH = 4
GRID_HEIGHT = 3
GRID = [
    [0, 0, 0, 1],
    [0, 0, 0, -1],
    [0, 0, 0, 0]
]

OBSTACLES = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0]
]

# Harder problem
# GRID_WIDTH = 50
# GRID_HEIGHT = 50
# GRID = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
#
# GRID[0][40] = -1
# GRID[30][45] = 1
#
# OBSTACLE_PROB = 0.1
# OBSTACLES = [[[0, 1][random() < OBSTACLE_PROB and not GRID[y][x]]
#               for x in range(GRID_WIDTH)]
#              for y in range(GRID_HEIGHT)]

ACTIONS = {"UP": (0, 1), "RIGHT": (1, 0), "DOWN": (0, -1), "LEFT": (-1, 0)}

TRAIN_EPSILON = 0.8
TEST_EPSILON = 0.0
DISCOUNT = 0.9
ALPHA = 0.8
DISPLACEMENT_COST = 0.1
START_POS = (0, 2)

PROB_MOV_SUCCESS = 0.8
MOV_FAIL_PROBS = {"LEFT": 0.1, "RIGHT": 0.1}

NB_TRAINING_ROUNDS = 10000
NB_TESTING_ROUNDS = 1000


def get_reward(state):
    return GRID[state[1]][state[0]] - DISPLACEMENT_COST


def create_q_table():
    return defaultdict(int)


def create_agent(start_state):
    return start_state, create_q_table()


def random_move_fail(random_value, act_state):
    new_value = random_value - PROB_MOV_SUCCESS
    for action, prob in MOV_FAIL_PROBS.items():
        if new_value < prob and legal_state(get_next_state(act_state, action)):
            return action
        new_value -= prob
    return None


def get_failable_next_state(state, action):
    random_value = random()
    real_action = action if random_value < PROB_MOV_SUCCESS else random_move_fail(random_value, state)
    return get_next_state(state, real_action)


def get_next_state(state, action):
    return tuple(state[i] + ACTIONS[action][i] for i in range(2))


def legal_state(state):
    return 0 <= state[0] < GRID_WIDTH \
           and 0 <= state[1] < GRID_HEIGHT \
           and not OBSTACLES[state[1]][state[0]]


def get_q_value(q_table, state, action):
    return q_table[(state, action)]


def get_legal_actions(state):
    return [key for key, action in ACTIONS.items() if legal_state(get_next_state(state, key))]


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
    return GRID[state[1]][state[0]] != 0


def explore(agent):
    state, q_table = agent
    while not final_state(state):
        action = get_next_action(q_table, state, TRAIN_EPSILON)
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
        action = get_next_action(q_table, state, TEST_EPSILON)
        state = get_next_state(state, action)

    return state, nb_turn


def main():
    agent = create_agent(START_POS)

    train_successes = 0
    # NB SUCCESS - TOTAL NB TURN
    test_stats = [0, 0]

    for i in range(NB_TRAINING_ROUNDS):
        final_train_state = explore(agent)
        if GRID[final_train_state[1]][final_train_state[0]] == 1:
            train_successes += 1

    for i in range(NB_TESTING_ROUNDS):
        final_test_state, nb_turn = exploit(agent)
        if GRID[final_test_state[1]][final_test_state[0]] == 1:
            test_stats[0] += 1
        test_stats[1] += nb_turn

    print(f"Train Success : {train_successes / NB_TRAINING_ROUNDS:.2%}\n"
          f"Test Success : {test_stats[0] / NB_TESTING_ROUNDS:.2%}\n"
          f"Mean Nb Turn : {test_stats[1] / NB_TESTING_ROUNDS:.0f}\n")


main()
