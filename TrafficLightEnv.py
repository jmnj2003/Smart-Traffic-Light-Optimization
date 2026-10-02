import numpy as np
import pygame
import sys
import time
import matplotlib.pyplot as plt
class TrafficLightEnv:
    def __init__(self):
        self.states = {0: 'Normal (N)', 1: 'Fault (F)'}
        self.actions = {0: 'Change (C)', 1: 'Attempt Fix (A)'}
        self.colors = {0: 'Red', 1: 'Green', 2: 'Yellow'}
        self.state = 0
        self.color = 0
        self.last_action = None

    def reset(self):
        self.state = 0
        self.color = 0
        self.last_action = None
        return self.state

    def step(self, action):
        self.last_action = action
        if self.state == 0:  # Normal State
            if action == 0:  # Change Action
                self.color = (self.color + 1) % 3
                if np.random.rand() < 0.05:
                    self.state = 1
                else:
                    self.state = 0
                reward = -1
            else:  # Attempt Fix duting normal state
                if np.random.rand() < 0.5:
                    self.state = 1
                else:
                    self.state = 0
                reward = -5
        else:  # Fault State
            if action == 0:
                self.state = 1
                reward = -5
            else:
                self.state = 0 if np.random.rand() < 0.8 else 1
                if self.state == 0:
                    self.color = 0
                reward = -5

        return self.state, reward, False


def train_agent(env, episodes=2000, steps_per_episode=80, alpha=0.12, gamma=0.96, epsilon=0.18):
    q_table = np.zeros((2, 2))
    episode_rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(steps_per_episode):
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2)
            else:
                action = np.argmax(q_table[state])
            next_state, reward, _ = env.step(action)
            best_next = np.argmax(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next] - q_table[state, action])
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
    return q_table, episode_rewards


def visual_simulate(env, q_table, steps=400, fps=5, delay=1):
    pygame.init()
    screen = pygame.display.set_mode((440, 780))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 40)

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    YELLOW_BRIGHT = (255, 255, 100)
    YELLOW_DIM = (140, 140, 0)
    GREEN = (0, 255, 0)
    GRAY = (70, 70, 70)

    state = env.reset()
    cum_reward = 0
    blink = False
    blink_timer = 0

    for step in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if state == 1:
            action = 1
        else:
            action = np.argmax(q_table[state])

        next_state, reward, _ = env.step(action)
        cum_reward += reward
        state = next_state

        if step % 60 in [15, 16, 17, 18, 19, 20]:
            env.state = 1

        if state == 1:
            blink_timer += 1
            if blink_timer % 6 == 0:
                blink = not blink

        screen.fill(BLACK)
        pygame.draw.rect(screen, (60, 60, 60), (195, 90, 50, 340))

        light_colors = [GRAY, GRAY, GRAY]

        if state == 0:
            if env.color == 0: light_colors[0] = RED
            elif env.color == 1: light_colors[1] = YELLOW_BRIGHT
            elif env.color == 2: light_colors[2] = GREEN
        else:
            current_yellow = YELLOW_BRIGHT if blink else YELLOW_DIM
            light_colors[1] = current_yellow

        pygame.draw.circle(screen, light_colors[0], (220, 150), 48)
        pygame.draw.circle(screen, light_colors[1], (220, 260), 48)
        pygame.draw.circle(screen, light_colors[2], (220, 370), 48)

        texts = [
            f"State: {env.states[state]}",
            f"Action: {env.actions[action]}",
            f"Reward: {reward}",
            f"Total: {cum_reward}"
        ]

        for i, text in enumerate(texts):
            surf = font.render(text, True, WHITE)
            screen.blit(surf, (30, 540 + i * 52))

        pygame.display.flip()
        time.sleep(delay)
        clock.tick(fps)

    pygame.quit()

def plot_learning_curve(episode_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, color='royalblue', alpha=0.6, label='Episode reward')

    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg,
                 color='darkred', linewidth=2.5, label=f'{window}-episode moving avg')

    plt.title("Learning Curve - Cumulative Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = TrafficLightEnv()
    q_table, episode_rewards = train_agent(env)

    print("\nFinal Q-table:")
    print("           Change    Fix")
    print(f"Normal  {q_table[0,0]:6.2f}   {q_table[0,1]:6.2f}")
    print(f"Fault   {q_table[1,0]:6.2f}   {q_table[1,1]:6.2f}\n")

    # Show learning curve
    plot_learning_curve(episode_rewards)
    visual_simulate(env, q_table)
