import torch
import pygame
from dinoGame import DinoGame, DQNAgent  # Importa tu agente DQN

def load_and_play():
    # Cargar el juego y el modelo
    env = DinoGame()
    state_size = 9
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    # Cargar los pesos del modelo
    agent.model.load_state_dict(torch.load('best_dino_model.pth'))
    agent.model.eval()  # Poner el modelo en modo evaluación

    # Configurar sin exploración
    agent.epsilon = 0.0

    # Jugar con el modelo cargado
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        env.render()  # Mostrar el juego en pantalla
        total_reward += reward

    print(f"Partida finalizada. Puntaje: {total_reward}")
    pygame.quit()

if __name__ == "__main__":
    load_and_play()
