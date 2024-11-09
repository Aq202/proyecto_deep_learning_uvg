import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DinoGame:
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 200
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Dino Game')
        
        # Configuración del dinosaurio
        self.dino_height = 40
        self.dino_width = 20
        self.dino_x = 50
        self.dino_y = self.height - self.dino_height
        self.dino_vel = 0
        self.jump_vel = -15
        self.gravity = 0.8
        
        # Configuración de obstáculos
        self.obstacle_width = 20
        self.obstacle_height = 50
        self.obstacle_x = self.width
        self.obstacle_speed = 5
        
        self.score = 0
        self.game_over = False
        self.clock = pygame.time.Clock()
        self.paused = False
        
    def handle_events(self):
        pygame.event.pump()  # Procesa los eventos en segundo plano
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
        return True
    
    def reset(self):
        self.dino_y = self.height - self.dino_height
        self.dino_vel = 0
        self.obstacle_x = self.width
        self.score = 0
        self.game_over = False
        self.paused = False
        return self._get_state()
    
    def _get_state(self):
        return np.array([
            self.obstacle_x - self.dino_x,
            self.dino_y,
            self.dino_vel
        ])
    
    def step(self, action):
        if not self.handle_events() or self.paused:
            return self._get_state(), 0, True

        reward = 0.1
        
        if action == 1 and self.dino_y >= self.height - self.dino_height:
            self.dino_vel = self.jump_vel
        
        self.dino_y += self.dino_vel
        self.dino_vel += self.gravity
        
        if self.dino_y > self.height - self.dino_height:
            self.dino_y = self.height - self.dino_height
            self.dino_vel = 0
        
        self.obstacle_x -= self.obstacle_speed
        if self.obstacle_x < -self.obstacle_width:
            self.obstacle_x = self.width
            reward = 1.0
            self.score += 1
        
        if self._check_collision():
            reward = -10.0
            self.game_over = True
        
        return self._get_state(), reward, self.game_over
    
    def _check_collision(self):
        dino_rect = pygame.Rect(self.dino_x, self.dino_y, self.dino_width, self.dino_height)
        obstacle_rect = pygame.Rect(self.obstacle_x, self.height - self.obstacle_height,
                                self.obstacle_width, self.obstacle_height)
        return dino_rect.colliderect(obstacle_rect)
    
    def render(self):
        self.screen.fill((255, 255, 255))
        
        # Dibujar dinosaurio
        pygame.draw.rect(self.screen, (0, 0, 0),
                        (self.dino_x, self.dino_y, self.dino_width, self.dino_height))
        
        # Dibujar obstáculo
        pygame.draw.rect(self.screen, (0, 0, 0),
                        (self.obstacle_x, self.height - self.obstacle_height,
                        self.obstacle_width, self.obstacle_height))
        
        # Dibujar puntuación
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        
        if self.paused:
            pause_text = font.render('PAUSED', True, (255, 0, 0))
            text_rect = pause_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(pause_text, text_rect)
        
        pygame.display.flip()
        self.clock.tick(60)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=10000)
        
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dino():
    env = DinoGame()
    state_size = 3
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000
    
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            while not env.game_over:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                if done and not env.paused:  # Si terminó por colisión
                    total_reward += reward
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay(batch_size)
                elif not env.paused:  # Si no está pausado
                    total_reward += reward
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay(batch_size)
                
                state = next_state
                env.render()
                
                if episode % 10 == 0:
                    agent.update_target_model()
            
            print(f"Episode: {episode + 1}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
    finally:
        pygame.quit()

if __name__ == "__main__":
    train_dino()