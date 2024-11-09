import numpy as np
import pygame


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


if __name__ == "__main__":
    game = DinoGame()
    state = game.reset()
    
    while True:
        action = np.random.choice([0, 1])
        next_state, reward, done = game.step(action)
        game.render()
        
        if done:
            state = game.reset()
        else:
            state = next_state