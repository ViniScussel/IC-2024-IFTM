import random
import copy
import gymnasium as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# Criando o ambiente
env = gym.make('FrozenLake-v1', is_slippery=False)
state_dims = env.observation_space.n  # O espaço de observação é discreto
num_actions = env.action_space.n
print(f"FrozenLake env: State dimensions: {state_dims}, Number of actions: {num_actions}")

save_path = "model_checkpoint.pth"

class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs, _ = self.env.reset()
        return self.one_hot(obs)

    def step(self, action):
        action = action.item()
        next_state, reward, done, _, info = self.env.step(action)
        next_state = self.one_hot(next_state)
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1).float()
        return next_state, reward, done, info

    def one_hot(self, state):
        one_hot_state = torch.zeros(state_dims)
        one_hot_state[state] = 1.0  # Ativa a posição correspondente ao estado
        return one_hot_state.unsqueeze(0).float()  # Adiciona uma dimensão para batch

env = PreprocessEnv(env)

# Função para obter um estado inicial
state = env.reset()
action = torch.tensor(0)
next_state, reward, done, _ = env.step(action)
print(f"Sample state: {state}")
print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")

# Definindo a rede neural
q_network = nn.Sequential(
    nn.Linear(state_dims, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, num_actions)
)

target_q_network = copy.deepcopy(q_network).eval()

def policy(state, epsilon=0.):
    if torch.rand(1) < epsilon:
        return torch.randint(num_actions, (1, 1))
    else:
        av = q_network(state).detach()
        return torch.argmax(av, dim=-1, keepdim=True)

class ReplayMemory:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)

def deep_sarsa(q_network, policy, episodes, alpha=0.001,
               batch_size=32, gamma=0.99, epsilon=0.05):
    global env
    optim = AdamW(q_network.parameters(), lr=alpha)
    memory = ReplayMemory()
    stats = {'MSE Loss': [], 'Returns': []}

    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return = 0
        while not done:
            action = policy(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            memory.insert([state, action, reward, done, next_state])

            if memory.can_sample(batch_size):
                state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)
                qsa_b = q_network(state_b).gather(1, action_b)
                next_action_b = policy(next_state_b, epsilon)
                next_qsa_b = target_q_network(next_state_b).gather(1, next_action_b)
                target_b = reward_b + (1 - done_b) * gamma * next_qsa_b
                loss = F.mse_loss(qsa_b, target_b)
                q_network.zero_grad()
                loss.backward()
                optim.step()
                stats['MSE Loss'].append(loss.item())

            state = next_state
            ep_return += reward.item()

        stats['Returns'].append(ep_return)

        if episode % 100 == 0:
            torch.save(q_network.state_dict(), save_path)
            print(f"Modelo salvo no episódio {episode}")

        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

    # Plotar recompensas por episódio
    plt.figure(figsize=(10, 5))
    plt.plot(stats['Returns'], label='Recompensa por Episódio', color='blue')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.title('G_0/Ep Deep SARSA')
    plt.legend()
    plt.grid()
    plt.show()

    return stats

# Executar o treinamento
stats = deep_sarsa(q_network, policy, 1000, epsilon=0.01)

# Você pode plotar novamente após o treinamento, caso queira
# plt.figure(figsize=(10, 5))
# plt.plot(stats['Returns'], label='Recompensa por Episódio', color='blue')
# plt.xlabel('Episódio')
# plt.ylabel('Recompensa Total')
# plt.title('Recompensas por Episódio no Treinamento')
# plt.legend()
# plt.grid()
# plt.show()