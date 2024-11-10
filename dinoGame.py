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
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Obstacle:
    def __init__(self, x, height, width, is_bird=False):
        self.x = x
        self.height = height
        self.width = width
        self.is_bird = is_bird
        self.y = 0 if is_bird else None

class DinoGame:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 300
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Dino Game')
        
        # Dinosaur configuration
        self.dino_height = 40
        self.dino_width = 20
        self.dino_x = 50
        self.dino_y = self.height - self.dino_height
        self.dino_vel = 0
        self.jump_vel = -15
        self.gravity = 0.8
        self.is_ducking = False
        self.duck_height = self.dino_height // 2
        
        # Obstaculos
        self.obstacles = []
        self.bird_heights = [self.height - 120, self.height - 80] # Altura permitida de pájaros
        
        # Espacio mínimo entre obstáculos
        self.absolute_min_spacing = 400
        self.last_obstacle_x = None
        self.max_obstacles = 2
        
        # Dificultad del juego
        self.base_speed = 5
        self.current_speed = self.base_speed
        self.speed_increment = 0.05
        self.max_speed = 12
        
        self.score = 0
        self.game_over = False
        self.clock = pygame.time.Clock()
        self.paused = False
    
    def handle_events(self):
        pygame.event.pump()
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
        self.obstacles = []
        self.score = 0
        self.game_over = False
        self.paused = False
        self.current_speed = self.base_speed
        self.is_ducking = False
        self.last_obstacle_x = None
        return self._get_state()
    
    def _get_state(self):
        closest_obstacles = sorted(self.obstacles, key=lambda x: x.x)[:2]
        state = []
        
        # Añadir información de los dos obstáculos más cercanos
        for i in range(2):
            if i < len(closest_obstacles):
                obs = closest_obstacles[i]
                state.extend([
                    obs.x - self.dino_x,
                    obs.height,
                    1 if obs.is_bird else 0,
                    obs.y if obs.is_bird else self.height - obs.height
                ])
            else:
                state.extend([self.width, 0, 0, 0])
        
        # Agregar stado del dinosaurio
        state.extend([
            self.dino_y,
            self.dino_vel,
            1 if self.is_ducking else 0,
            self.current_speed
        ])
        
        return np.array(state)
    
    def _can_spawn_obstacle(self):
        if len(self.obstacles) >= self.max_obstacles:
            return False
            
        if not self.obstacles:
            return True if self.last_obstacle_x is None else False
            
        rightmost_x = max(obstacle.x for obstacle in self.obstacles)
        required_spacing = self.absolute_min_spacing + (self.current_speed * 20)
        
        return rightmost_x <= (self.width - required_spacing)
    
    def _spawn_obstacle(self):
        if not self._can_spawn_obstacle():
            return
            
        is_bird = random.random() < 0.2
        
        if is_bird:
            height = 30
            width = 30
            obstacle = Obstacle(self.width, height, width, is_bird=True)
            obstacle.y = random.choice(self.bird_heights)
        else:
            height = random.randint(30, 60)
            width = 20
            obstacle = Obstacle(self.width, height, width, is_bird=False)
        
        self.obstacles.append(obstacle)
        self.last_obstacle_x = self.width
    
    def step(self, action):
        if not self.handle_events() or self.paused:
            return self._get_state(), 0, True

        reward = 0.1
        
        if action == 1 and self.dino_y >= self.height - self.dino_height:
            self.dino_vel = self.jump_vel
            self.is_ducking = False
        elif action == 2 and self.dino_y >= self.height - self.dino_height:
            self.is_ducking = True
        else:
            self.is_ducking = False
        
        self.dino_y += self.dino_vel
        self.dino_vel += self.gravity
        
        if self.dino_y > self.height - (self.duck_height if self.is_ducking else self.dino_height):
            self.dino_y = self.height - (self.duck_height if self.is_ducking else self.dino_height)
            self.dino_vel = 0
        
        if len(self.obstacles) > 0:
            self.current_speed = min(self.max_speed, 
                                   self.base_speed + self.speed_increment * self.score)
        
        any_obstacle_passed = False
        for obstacle in self.obstacles[:]:
            obstacle.x -= self.current_speed
            if obstacle.x < -obstacle.width:
                self.obstacles.remove(obstacle)
                any_obstacle_passed = True
        
        if any_obstacle_passed:
            self.score += 1
            reward = 1.0
        
        if self._can_spawn_obstacle():
            self._spawn_obstacle()
        
        if self._check_collision():
            reward = -10.0
            self.game_over = True
        
        return self._get_state(), reward, self.game_over
    
    def _check_collision(self):
        dino_height = self.duck_height if self.is_ducking else self.dino_height
        dino_rect = pygame.Rect(self.dino_x, self.dino_y, self.dino_width, dino_height)
        
        for obstacle in self.obstacles:
            if obstacle.is_bird:
                obstacle_rect = pygame.Rect(obstacle.x, obstacle.y,
                                            obstacle.width, obstacle.height)
            else:
                obstacle_rect = pygame.Rect(obstacle.x,
                                            self.height - obstacle.height,
                                            obstacle.width, obstacle.height)
            
            if dino_rect.colliderect(obstacle_rect):
                return True
        return False
    
    def render(self):
        self.screen.fill((255, 255, 255))
        
        # Dibujar al dinosaurio
        dino_height = self.duck_height if self.is_ducking else self.dino_height
        pygame.draw.rect(self.screen, (0, 0, 0),
                        (self.dino_x, self.dino_y, self.dino_width, dino_height))
        
        # Dibujar los obstáculos
        for obstacle in self.obstacles:
            if obstacle.is_bird:
                pygame.draw.rect(self.screen, (255, 0, 0),
                                (obstacle.x, obstacle.y,
                                obstacle.width, obstacle.height))
            else:
                pygame.draw.rect(self.screen, (0, 0, 0),
                                (obstacle.x, self.height - obstacle.height,
                                obstacle.width, obstacle.height))
        
        # Dibujar labels de score y velocidad
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (0, 0, 0))
        speed_text = font.render(f'Speed: {self.current_speed:.1f}', True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(speed_text, (10, 40))
        
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
        self.memory = deque(maxlen=100000000)
        
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
    state_size = 12
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    batch_size = 128
    episodes = 1000
    
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            while not env.game_over:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                if done and not env.paused:
                    total_reward += reward
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay(batch_size)
                elif not env.paused:
                    total_reward += reward
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay(batch_size)
                
                state = next_state
                env.render()
                
                if episode % 10 == 0:
                    agent.update_target_model()
            
            print(f"Episodio: {episode + 1}, Score: {total_reward}, Epsilon: {agent.epsilon}")
    
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
    finally:
        pygame.quit()

if __name__ == "__main__":
    train_dino()