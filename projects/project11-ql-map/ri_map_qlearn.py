import random
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import mapgen as mg
import mapcontroller as mc
from mapcontroller import MoveResult


random.seed(32)

ACTIONS = ["left", "right", "up", "down"]

ALPHA           = 0.1     # learning rate
GAMMA           = 0.95    # discount factor
EPSILON_START   = 1.0     # fully random at first
EPSILON_END     = 0.05    # keep a little exploration forever
EPSILON_DECAY   = 0.995   # per episode
EPISODES        = 2000
MAX_STEPS       = 500

REWARD_EAT      =  10.0
REWARD_CLEAR    =  50.0
REWARD_WALL     =  -1.0
REWARD_OOB      =  -1.0
REWARD_EMPTY    =  -1.0
REWARD_STEP     =  -0.05


COLORS = ['#1a1a2e', '#4a4a5a', '#c8a96e', '#00e676', '#ff1744']
LABELS = ['Empty', 'Wall', 'Floor', 'Player', 'Enemy']


def display_map(grid, title="Map"):
    cmap = ListedColormap(COLORS)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(COLORS) - 1,
              interpolation='nearest', aspect='equal')
    legend = [mpatches.Patch(color=COLORS[i], label=f'{i} – {LABELS[i]}')
              for i in range(len(COLORS))]
    ax.legend(handles=legend, loc='upper right', fontsize=9,
              framealpha=0.85, edgecolor='#888')
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def encode_state(controller):
    """State = (player position, frozenset of remaining enemy positions).

    Including the enemy set matters: 'player at (5,5) with 3 enemies left' is a
    different situation than 'player at (5,5) with all enemies eaten', and the
    best action may differ.
    """
    return (controller.player_position(),
            frozenset(controller.enemy_positions()))


def apply_action(controller, action):
    if action == "left":  return controller.move_player_left()
    if action == "right": return controller.move_player_right()
    if action == "up":    return controller.move_player_up()
    if action == "down":  return controller.move_player_down()
    raise ValueError(action)


def reward_for(result):
    if result == MoveResult.ATE_ENEMY:     return REWARD_EAT
    if result == MoveResult.MOVED:         return REWARD_STEP
    if result == MoveResult.HIT_WALL:      return REWARD_WALL
    if result == MoveResult.OUT_OF_BOUNDS: return REWARD_OOB
    if result == MoveResult.HIT_EMPTY:     return REWARD_EMPTY
    return 0.0


def choose_action(Q, state, epsilon):
    """Epsilon-greedy: random sometimes, otherwise best known action."""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    values = Q[state]
    best = max(values.values())
    best_actions = [a for a, v in values.items() if v == best]
    return random.choice(best_actions)


def train(original_grid):
    # defaultdict so any unseen state starts with all-zero Q-values
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    epsilon = EPSILON_START

    cleared_count = 0
    for ep in range(EPISODES):
        controller = mc.MapController(original_grid)
        state = encode_state(controller)

        for _ in range(MAX_STEPS):
            action = choose_action(Q, state, epsilon)
            result = apply_action(controller, action)
            reward = reward_for(result)

            done = controller.all_enemies_eaten()
            if done:
                reward += REWARD_CLEAR

            next_state = encode_state(controller)

            # Bellman update: shift Q[s][a] toward (reward + discounted best future value)
            old_value = Q[state][action]
            future    = 0.0 if done else max(Q[next_state].values())
            Q[state][action] = old_value + ALPHA * (reward + GAMMA * future - old_value)

            state = next_state
            if done:
                cleared_count += 1
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (ep + 1) % 100 == 0:
            print(f"episode {ep + 1:4d} | epsilon {epsilon:.3f} "
                  f"| cleared {cleared_count}/100 in last window "
                  f"| |Q| = {len(Q)}")
            cleared_count = 0

    return Q


def run_greedy(original_grid, Q, render=True):
    """Play one episode using the learned policy, no exploration."""
    controller = mc.MapController(original_grid)
    if render:
        display_map(controller.snapshot(), "Start")

    steps = 0
    while not controller.all_enemies_eaten() and steps < MAX_STEPS:
        state = encode_state(controller)
        # If we've never seen this state, fall back to random
        if state not in Q:
            action = random.choice(ACTIONS)
        else:
            values = Q[state]
            action = max(values, key=values.get)
        apply_action(controller, action)
        steps += 1

    print(f"greedy rollout: cleared={controller.all_enemies_eaten()} steps={steps}")
    if render:
        display_map(controller.snapshot(), f"End (steps={steps})")


if __name__ == "__main__":
    game_map_data = mg.Cellular().generate(32, 32, 0.6, 10)
    print(f"enemies on map: {mc.MapController(game_map_data).enemy_count()}")

    Q = train(game_map_data)
    run_greedy(game_map_data, Q)
