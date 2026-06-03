"""
Use q-learning to play blackjack

Q-learning formula: Qopt(s, a) <- (1 - lr) * Qopt(s,a) + lr * (r + Vopt(s'))
Vopt(s') = max {a' in actions(s')} (Qopt(s', a'))
"""

import numpy as np
from collections import deque

def card(): # Using an infinite deck for simplicity
    value = np.random.randint(1,14)

    return min(value, 10)


# Hyperparameters
gamma = 1
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay = 0.99995
lr = 0.001
episodes = 100000

length = 1000
wins = deque(maxlen=length)
draws = deque(maxlen=length)
losses = deque(maxlen=length)

q_table = np.zeros((18, 10, 2, 2)) # Initialize the q-table [hand_value (4-21), dealer_show_card (1-10), playeable ace (0,1), hit/stick (0,1)]
epsilon = epsilon_start

# 0 = hit, 1 = stick

print("TRAINING")

for i in range(episodes):
    finished = False
    reward = 0
    total = 0
    ace = 0
    d_ace = 0
    dealer = card() # Draw show card for the dealer

    for j in range(2): # Draw two cards for the player
        draw = card()
        if draw == 1:
            if total < 11:
                ace = 1
                draw = 11
        total += draw

    while not finished:
        if np.random.random() < epsilon:
            move = np.random.randint(0,2)
        else:
            state_qs = q_table[total-4, dealer-1, ace] # Select a move
            if state_qs[0] == state_qs[1]:
                move = np.random.randint(0, 2)
            else:
                move = np.argmax(state_qs)

        if move == 0: # If hit
            prev_tot = total
            prev_ace = ace
            draw = card() # Draw card
            if draw == 1:
                if total < 11:
                    ace = 1
                    draw = 11
            total += draw
            
            if total > 21: # Check if it goes overboard, switch ace value if needed
                if ace:
                    total -= 10
                    ace = 0
                    target = reward + gamma * np.max(q_table[total-4, dealer-1, ace])
                else:
                    reward = -1
                    finished = True
                    target = reward
            else:
                target = reward + gamma * np.max(q_table[total-4, dealer-1, ace])

            q_table[prev_tot-4, dealer-1, prev_ace, move] = (1-lr) * q_table[prev_tot-4, dealer-1, prev_ace, move] + lr * target # Update q-table
            

        else: # If stick

            # Play the dealer
            prev_dealer = dealer
            if dealer == 1:
                dealer = 11
                d_ace = 1
            draw = card()
            if draw == 1:
                if dealer < 11:
                    d_ace = 1
                    draw = 11
            dealer += draw

            while dealer <= 16:
                draw = card()
                if draw == 1:
                    if dealer < 11:
                        d_ace = 1
                        draw = 11
                dealer += draw
                if dealer > 21:
                    if d_ace:
                        dealer -= 10
                        d_ace = 0
            
            if dealer > 21:
                if d_ace:
                    dealer -= 10
                    d_ace = 0

            if dealer > 21 or total > dealer: # Find a winner
                reward = 1
            elif dealer == total:
                reward = 0
            else:
                reward = -1

            finished = True

            q_table[total-4, prev_dealer-1, ace, move] = (1-lr) * q_table[total-4, prev_dealer-1, ace, move] + lr * reward

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    wins.append(1 if reward == 1 else 0)
    draws.append(1 if reward == 0 else 0)
    losses.append(1 if reward == -1 else 0)

    if i % 1000 == 0:
        print(f"Win rate: {sum(wins)/len(wins)}, Draw rate: {sum(draws)/len(draws)}, Loss rate: {sum(losses)/len(losses)}")
 
print()
print("-------------------------------------------------")
print("TESTING")

tests = 1000000

won = 0
drawed = 0
lost = 0

for i in range(tests): # Test loop, same thing as training but no randomness from epsilon and no updating the table
    finished = False
    reward = 0
    total = 0
    ace = 0
    d_ace = 0
    dealer = card()

    for j in range(2):
        draw = card()
        if draw == 1:
            if total < 11:
                ace = 1
                draw = 11
        total += draw

    while not finished:
        state_qs = q_table[total-4, dealer-1, ace]
        if state_qs[0] == state_qs[1]:
            move = np.random.randint(0, 2)
        else:
            move = np.argmax(state_qs)

        if move == 0:
            prev_tot = total
            prev_ace = ace
            draw = card()
            if draw == 1:
                if total < 11:
                    ace = 1
                    draw = 11
            total += draw
            
            if total > 21:
                if ace:
                    total -= 10
                    ace = 0
                else:
                    lost += 1
                    finished = True
            

        else:
            prev_dealer = dealer
            if dealer == 1:
                dealer = 11
                d_ace = 1
            draw = card()
            if draw == 1:
                if dealer < 11:
                    d_ace = 1
                    draw = 11
            dealer += draw

            while dealer <= 16:
                draw = card()
                if draw == 1:
                    if dealer < 11:
                        d_ace = 1
                        draw = 11
                dealer += draw
                if dealer > 21:
                    if d_ace:
                        dealer -= 10
                        d_ace = 0
            
            if dealer > 21:
                if d_ace:
                    dealer -= 10
                    d_ace = 0

            if dealer > 21 or total > dealer:
                won += 1
            elif dealer == total:
                drawed += 1
            else:
                lost += 1

            finished = True

print(f"Win rate: {won/tests}")
print(f"Draw rate: {drawed/tests}")
print(f"Loss rate: {lost/tests}")