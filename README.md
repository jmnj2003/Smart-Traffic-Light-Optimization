# Smart Traffic Light Optimization with Q-Learning

A reinforcement learning agent that learns how to run a traffic light controller: keep the normal Red → Green → Yellow cycle going, and repair the light quickly when it breaks. The agent is trained with tabular **Q-learning** and the learned policy is shown in a small **Pygame** simulation.

> Group project for **SAIA 2123 Reinforcement Learning**, Bachelor in Artificial Intelligence, Universiti Teknologi Malaysia (Semester 1, 2025/2026).

<p align="center">
  <img src="assets/poster.jpg" alt="Project poster: Smart Traffic Light Optimization" width="560">
</p>

---

## Problem

Traditional traffic lights run on fixed, preset cycles. They can't adapt when conditions change, for example when a signal gets stuck or starts blinking. A faulty light that isn't fixed quickly causes delays and accidents and costs money.

**Goal:** train an RL agent that prioritises keeping normal signal cycles running while recovering from faults efficiently.

## Environment design (MDP)

| Component | Definition |
|---|---|
| **Agent** | The traffic light controller |
| **States** | `0` Normal Operation (N): the standard light cycle · `1` Fault (F): the signal is stuck or blinking |
| **Actions** | `0` Change Light (C): advance to the next colour · `1` Attempt Fix (A): try to repair the signal |
| **Rewards** | `-1` for a normal Change · `-5` for any action in a fault state, or an unnecessary fix |

Transitions are stochastic:

| From | Action | Outcome | Reward |
|---|---|---|---|
| Normal | Change | 95% stay Normal · 5% become Fault | -1 |
| Normal | Attempt Fix (unnecessary) | 50% stay Normal · 50% become Fault | -5 |
| Fault | Change | Always stays Fault | -5 |
| Fault | Attempt Fix | 80% back to Normal · 20% stay Fault | -5 |

Every reward is negative, so the agent learns to **minimise long-term penalties**.

## Method: Q-learning

Q-learning is **model-free**: the agent never sees the transition probabilities above and learns only from experience. With only 2 states × 2 actions, a 2×2 Q-table covers every case.

Update rule after each step:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

Actions are chosen with an **ε-greedy** policy: a random action with probability ε (exploration), otherwise the best-known action (exploitation).

| Hyperparameter | Value |
|---|---|
| Episodes | 2,000 |
| Steps per episode | 80 |
| Learning rate α | 0.12 |
| Discount factor γ | 0.96 |
| Exploration rate ε | 0.18 (fixed) |

<details>
<summary>Training flowchart</summary>
<p align="center"><img src="assets/training_flowchart.png" alt="Q-learning training flowchart" width="320"></p>
</details>

## Results

<p align="center">
  <img src="assets/learning_curve.png" alt="Cumulative reward per episode over 2,000 episodes with 100-episode moving average" width="760">
</p>

- **Learned policy:** *Change Light* in the Normal state and *Attempt Fix* in the Fault state, which is the intended behaviour.
- **Convergence:** the 100-episode moving average levels off at about **−140 reward per episode** and stays there, so the Q-values have stabilised.
- **Variance:** episode rewards still swing between about −100 and −220. This comes from the fixed ε = 0.18: about 1 in 5 actions is random, and the faults themselves are random.

Example final Q-table (values vary slightly between runs because the environment is random):

| | Change | Attempt Fix |
|---|---|---|
| **Normal** | **−32.6** | −37.9 |
| **Fault** | −39.8 | **−36.9** |

The higher value in each row (shown in bold) is the action the agent picks.

## Simulation

After training, `visual_simulate()` opens a Pygame window that runs the learned policy step by step. It shows the light, the current state, the action, the reward and the running total.

- In the **Normal** state the light cycles Red → Yellow → Green.
- In the **Fault** state the yellow light blinks until the agent fixes it.
- To demonstrate recovery, the simulation forces the light into a fault for steps 15 to 20 of every 60.

## Getting started

```bash
git clone https://github.com/jmnj2003/smart-traffic-light-qlearning.git
cd smart-traffic-light-qlearning
pip install -r requirements.txt
python TrafficLightEnv.py
```

The script will:
1. train the agent and print the final Q-table,
2. show the learning curve (close the plot window to continue),
3. open the Pygame simulation (close the window to exit).

## Project structure

```
├── TrafficLightEnv.py        # environment, Q-learning training, plot, Pygame simulation
├── requirements.txt
└── assets/
    ├── poster.jpg            # project poster
    ├── learning_curve.png    # reward per episode
    └── training_flowchart.png
```

## Limitations and future work

- **Simplified environment:** only two states. It doesn't model traffic volume, queues, or multiple intersections working together.
- **Rare events:** the agent has no notion of priority traffic such as ambulances.
- **Human behaviour:** unpredictable drivers (late acceleration on green, blocked lanes) aren't modelled.
- **Scaling:** a Q-table won't scale to continuous or high-dimensional states. A **Deep Q-Network (DQN)** or multi-agent RL would be the next step.
- **Tuning:** an ε that decays over training, plus tuning α and γ, should reduce late-training noise.

## Team

Ahmad Haziq · Aiman Wafi · Bong Xin Ting · **Jessie Moh Ngiik Jun** · Lavinia Mary

Lecturer: Dr. Norshaliza binti Kamaruddin, Section 1

## References

- GeeksforGeeks (2025). *Q-Learning in Reinforcement Learning.*
- Simplilearn (2025). *Q-Learning Guide: Begin with Reinforcement Learning Basics.*
- Murtadha, H. W., Abdul Manan, M. M. B., & Hamad, W. A. (2021). *The Relationship between Population Growth and Vehicle Growth in Malaysia: A Review.*
