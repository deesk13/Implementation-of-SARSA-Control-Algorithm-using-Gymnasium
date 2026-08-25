## Implementation-of-SARSA-Control-Algorithm-using-Gymnasium ##
## Aim

To implement the SARSA control algorithm using the Gymnasium FrozenLake-v1 environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

## Problem Statement

Implement the SARSA control algorithm in the Gymnasium FrozenLake-v1 environment. The agent should learn an action-value function using an epsilon-greedy policy and improve its ability to reach the goal while avoiding holes.

## Software Requirements
Python 3.x
Gymnasium
NumPy
Matplotlib
Jupyter Notebook / Google Colab / VS Code

Install the required libraries using:

pip install gymnasium numpy matplotlib
Environment Description

The FrozenLake-v1 environment is a grid-world reinforcement learning environment.

The default 4 × 4 map is:

S F F F
F H F H
F F F H
H F F G

Where:

S → Starting state
F → Frozen surface
H → Hole
G → Goal

The agent starts at state 0 and must reach state 15.

The available actions are:

0 → LEFT
1 → DOWN
2 → RIGHT
3 → UP

In this implementation:

is_slippery=False

is used so that the agent moves deterministically.

Theory

SARSA stands for:

State → Action → Reward → Next State → Next Action

SARSA is an on-policy Temporal Difference control algorithm.

The SARSA update rule is:

Q(S_t,A_t) ← Q(S_t,A_t)
             + α[R_(t+1)
             + γQ(S_(t+1),A_(t+1))
             - Q(S_t,A_t)]

Where:

Symbol	Meaning
S_t	Current state
A_t	Current action
R_(t+1)	Reward received after taking the current action
S_(t+1)	Next state
A_(t+1)	Next action selected by the current policy
α	Learning rate
γ	Discount factor
Q(s,a)	Action-value function

The important point in SARSA is that the update uses the actual next action selected by the policy.

Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability ε, the agent chooses a random action.

With probability 1 - ε, the agent chooses the action with the highest Q-value.

                random action       if random < ε
Action =
                best Q-value action  otherwise

The value of epsilon is gradually reduced during training so that the agent initially explores and later relies more on learned Q-values.

## Algorithm
```
Initialize the Q-table with zeros.
Initialize the exploration rate epsilon.
Reset the environment.
Select the initial action using the epsilon-greedy policy.
Take the selected action.
Observe the next state and reward.
Select the next action using the epsilon-greedy policy.
Update the Q-value using the SARSA equation.
Move to the next state and action.
Repeat until the episode terminates.
Reduce epsilon.
Repeat for all episodes.
Display the final Q-table.
Calculate the average reward.
Plot the learning curve.
```
## Python Program
```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
`


# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.05
epsilon_decay = 0.9995


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

q_table = np.zeros(
    (env.observation_space.n, env.action_space.n)
)


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy strategy.
    """

    if np.random.rand() < epsilon:
        # Exploration
        return env.action_space.sample()

    else:
        # Exploitation
        return np.argmax(q_table[state])


# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Select initial action
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Select next action
        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # -------------------------------------------------
        # SARSA Update
        # -------------------------------------------------

        q_table[state, action] = q_table[state, action] + alpha * (
            reward
            + gamma * q_table[next_state, next_action]
            - q_table[state, action]
        )

        # Move to next state and action
        state = next_state
        action = next_action

        # Store reward
        total_reward += reward

        # Stop if episode ends
        if terminated or truncated:
            break

    # Store episode reward
    episode_rewards.append(total_reward)

    # -------------------------------------------------
    # Epsilon Decay
    # -------------------------------------------------

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


print("Training completed!")


# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(q_table, 3))

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "SARSA Learning Curve - FrozenLake"
)

plt.grid(True)

plt.show()


# -------------------------------------------------
# Close Environment
# -------------------------------------------------

env.close()
```
## Output
<img width="452" height="51" alt="image" src="https://github.com/user-attachments/assets/8627412a-5104-445a-9ced-e1052bb7fcd4" />
<img width="887" height="306" alt="image" src="https://github.com/user-attachments/assets/0f718448-8b5d-498d-85a8-cebbc63c1e14" />
<img width="758" height="405" alt="image" src="https://github.com/user-attachments/assets/01ebef11-07c0-4cd5-b4ee-13df64816fe8" />





## Result

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned Q-values through repeated interaction with the environment using an epsilon-greedy policy.

## Inference

The experiment demonstrates that SARSA can learn an effective action-value function through trial and error. Initially, the agent performs random exploration because epsilon is high. As training progresses, epsilon decreases and the agent increasingly selects actions based on the learned Q-values.

The learning curve can be used to observe the improvement in the agent's performance over the training episodes.
