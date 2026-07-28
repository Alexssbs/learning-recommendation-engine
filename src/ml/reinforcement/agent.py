import numpy as np
import json
import os

class QLearningAgent:
    def __init__(self, num_states: int, num_actions: int, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Inicializamos la Tabla Q con ceros (8 estados x 4 acciones)
        self.q_table = np.zeros((num_states, num_actions))
        
    def choose_action(self, state: int) -> int:
        """Elige una acción usando la estrategia Epsilon-Greedy"""
        if np.random.uniform(0, 1) < self.epsilon:
            # Exploración: elige una acción aleatoria
            return np.random.choice(self.num_actions)
        else:
            # Explotación: elige la mejor acción conocida para este estado
            return np.argmax(self.q_table[state, :])
            
    def update(self, state: int, action: int, reward: float, next_state: int):
        """Actualiza la Tabla Q usando la ecuación de Bellman"""
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        
        self.q_table[state, action] += self.alpha * td_error
        
    def save_model(self, filepath: str):
        """Guarda la política (Q-table) para ser usada en producción (API)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.q_table.tolist(), f)
            
    def load_model(self, filepath: str):
        """Carga una política existente"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.q_table = np.array(json.load(f))
