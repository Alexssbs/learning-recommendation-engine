from environment import SmartLearningEnv
from agent import QLearningAgent
import os
import numpy as np

def train():
    env = SmartLearningEnv()
    agent = QLearningAgent(num_states=env.num_states, num_actions=env.num_actions)
    
    episodes = 5000
    print(f"Entrenando al agente por {episodes} episodios...")
    
    rewards = []
    
    for episode in range(episodes):
        # Seleccionamos un perfil de estudiante al azar para simular
        state = np.random.randint(0, env.num_states)
        
        # El agente elige una recomendación
        action = agent.choose_action(state)
        
        # Obtenemos el feedback (recompensa)
        reward = env.get_reward(state, action)
        rewards.append(reward)
        
        # Para este caso simplificado
        next_state = state 
        
        # El agente aprende de la experiencia
        agent.update(state, action, reward, next_state)
        
        # Decaimiento de la exploración (epsilon)
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.995

    print("\nEntrenamiento finalizado. Tabla Q resultante:")
    print(agent.q_table)
    
    # Guardar modelo localmente
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "q_table.json")
    
    agent.save_model(model_path)
    print(f"\nModelo RL guardado localmente en: {model_path}")
    
    # Integración con Databricks MLflow (Senior MLOps)
    try:
        import mlflow
        print("\nConectando a Databricks MLflow para registrar RL...")
        mlflow.set_tracking_uri("databricks")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/mlops_clustering_prod")
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="RL_Q_Learning_Training"):
            mlflow.log_param("episodes", episodes)
            mlflow.log_param("learning_rate", agent.alpha)
            mlflow.log_param("discount_factor", agent.gamma)
            
            # Métrica Interna: Recompensa promedio final
            avg_reward = np.mean(rewards[-100:])
            mlflow.log_metric("avg_reward_last_100", avg_reward)
            
            # Validación Externa (Senior Proxy Metric): Expert Agreement Rate
            # ¿Qué tan bien se alinea la política aprendida con el escenario ideal?
            optimal_rewards = sum(env.get_reward(s, np.argmax(agent.q_table[s, :])) for s in range(env.num_states))
            # Suponiendo que la recompensa ideal para cada estado es 1.0 (aprobado máximo)
            max_possible_reward = env.num_states * 1.0 
            agreement_rate = optimal_rewards / max_possible_reward
            
            mlflow.log_metric("expert_agreement_rate", agreement_rate)
            
            # Guardamos el JSON como artefacto en MLflow
            mlflow.log_artifact(model_path, "q_learning_policy")
            
            print(f"✅ ¡Modelo RL registrado en Databricks! (Avg Reward: {avg_reward:.3f} | Agreement Rate: {agreement_rate:.3f})")
    except Exception as e:
        print(f"\n⚠️ Advertencia: No se pudo conectar a Databricks MLflow ({e}).")

if __name__ == "__main__":
    train()
