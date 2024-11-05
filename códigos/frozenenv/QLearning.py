import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt  # Importando matplotlib para plotagem

# Definindo constantes
N_EPISODES = 1000

# Inicializando valores de ação
action_values = np.zeros(shape=(16, 4))

# Criando o ambiente
env = gym.make('FrozenLake-v1', is_slippery=False)

def policy(state, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))

def constant_alpha_mc(policy, action_values, episodes=N_EPISODES, gamma=0.99, alpha=0.1):
    global env
    rewards_per_episode = []  # Lista para armazenar recompensas por episódio
    for episode in range(1, episodes + 1):
        if episode == episodes - 5:
            env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)
        state, _ = env.reset()
        state = int(state)
        done = False
        transitions = []
        G = 0

        while not done:
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            transitions.append([state, action, reward])
            state = int(next_state)

        # Cálculo do retorno G
        G = 0
        for state_t, action_t, reward_t in reversed(transitions):
            G = reward_t + gamma * G
            action_values[state_t][action_t] += alpha * (G - action_values[state_t][action_t])
        
        rewards_per_episode.append(sum(reward for _, _, reward in transitions))  # Armazenando a recompensa total do episódio

    return rewards_per_episode  # Retornar a lista de recompensas por episódio

# Executar o método de Monte Carlo
rewards = constant_alpha_mc(policy, action_values)

# Plotar recompensas por episódio
plt.figure(figsize=(10, 5))
plt.plot(rewards, label='Recompensa Total por Episódio', color='blue')
plt.xlabel('Episódio')
plt.ylabel('Recompensa Total')
plt.title('G_0/Ep Q-Learning')
plt.legend()
plt.grid()
plt.show()
