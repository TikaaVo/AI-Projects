"""
Use q-learning to solve a path problem with a random chance of slipping.

Q-learning formula: Qopt(s, a) <- (1 - lr) * Qopt(s,a) + lr * (r + Vopt(s'))
Vopt(s') = max {a' in actions(s')} (Qopt(s', a'))
"""


import numpy as np

# Actions: 0 = right, 1 = down
def get_next_state(sx, sy, action):
    if action == 0:
        sx += 1
    elif action == 1:
        sy += 1
    
    return np.clip(sx, 0, 3), np.clip(sy, 0, 3)

grid = np.array([
    [0,   0, -50,   0], 
    [0,  10, -50,   0], 
    [0,   0,   0, -50], 
    [0, -50,   0, 100]
])

# Hyperparameters
slip = 0.1
epsilon = 1.0
epsilon_decay = 0.995
epsilon_end = 0.01
gamma = 1.0
lr = 0.1
episodes = 1000

q_table = np.zeros((4, 4, 2))

for episode in range(episodes):
    sx, sy = 0, 0
    tot_reward = 0
    finished = False
    
    while not finished:
        if np.random.random() < epsilon:
            intended_move = np.random.randint(0, 2)
        else:
            state_qs = q_table[sy, sx]
            if state_qs[0] == state_qs[1]:
                intended_move = np.random.randint(0, 2) # Play a move
            else:
                intended_move = np.argmax(state_qs)

        actual_move = intended_move
        if np.random.random() < slip: # Slip logic
            actual_move = np.random.randint(0, 2)

        sx1, sy1 = get_next_state(sx, sy, actual_move) # Move and get reward
        reward = grid[sy1, sx1]
        tot_reward += reward

        if (sx1 == 3 and sy1 == 3) or reward < 0: # End the game if you get to the end or fall in a hole
            finished = True
            target = reward
        else:
            target = reward + gamma * np.max(q_table[sy1, sx1])

        q_table[sy, sx, intended_move] = (1 - lr) * q_table[sy, sx, intended_move] + lr * target # Update q-table

        sx, sy = sx1, sy1

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f"Episode: {episode + 1:4d} | Total Reward: {tot_reward:4d} | Epsilon: {epsilon:.4f}")