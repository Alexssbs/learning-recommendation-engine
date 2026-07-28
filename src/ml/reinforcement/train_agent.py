from environment import SmartLearningEnv
from agent import QLearningAgent
import os

def train():
    env = SmartLearningEnv()
    agent = QLearningAgent(num_states=env.num_states, num_actions=env.num_actions)
    
    episodes = 5000
    print(f"Entrenando al agente por {episodes} episodios...")
    
    for episode in range(episodes):
        # Seleccionamos un perfil de estudiante al azar para simular
        state = np.random.randint(0, env.num_states)
        
        # El agente elige una recomendación
        action = agent.choose_action(state)
        
        # Obtenemos el feedback (recompensa)
        reward = env.get_reward(state, action)
        
        # Para este caso simplificado, asumimos que el estado no cambia inmediatamente
        # en un mismo "episodio" de recomendación, o pasa a un estado terminal.
        # (En un proyecto real de RL, la recomendación podría cambiar el perfil del alumno)
        next_state = state 
        
        # El agente aprende de la experiencia
        agent.update(state, action, reward, next_state)
        
        # Decaimiento de la exploración (epsilon)
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.995

    print("\nEntrenamiento finalizado. Tabla Q resultante:")
    print(agent.q_table)
    
    # Guardar modelo en la carpeta models
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "q_table.json")
    
    agent.save_model(model_path)
    print(f"\nModelo RL guardado exitosamente en: {model_path}")

if __name__ == "__main__":
    import numpy as np
    train()
