import numpy as np

class SmartLearningEnv:
    """
    Entorno simplificado para el Aprendizaje por Refuerzo (Q-Learning).
    Representa las interacciones con el estudiante.
    """
    def __init__(self):
        # 4 Perfiles de estudiante descubiertos por el Clustering:
        # 0: Constante, 1: Nocturno, 2: Riesgo de Abandono, 3: Intensivo
        self.num_states = 4 
        
        # 4 Recomendaciones posibles (Acciones):
        # 0: Video Resumen, 1: Cuestionario, 2: Descanso, 3: Alerta/Motivación
        self.num_actions = 4
        
    def get_reward(self, state: int, action: int) -> float:
        """
        Simulación de la recompensa basada en conocimiento pedagógico.
        En producción real, esto viene de la interacción del usuario en la web.
        """
        # Matriz de recompensas teóricas [Estado/Perfil][Acción]
        # Ej: Al perfil "Nocturno" (1) le va mal con videos largos a la madrugada (0)
        # pero bien con repasos rápidos (1)
        simulated_rewards = [
            [0.5, 1.0, 0.0, -0.5], # 0: Constante -> Le gustan los cuestionarios (1)
            [-1.0, 0.5, 1.0, 0.0], # 1: Nocturno -> Necesita descanso (2) o cuestionario corto
            [-0.5, -0.5, 0.0, 1.0],# 2: Riesgo Abandono -> Necesita Motivación (3)
            [1.0, 0.5, -1.0, -0.5] # 3: Intensivo -> Le gustan los videos largos (0)
        ]
        
        # Añadimos un poco de ruido aleatorio para simular el mundo real
        base_reward = simulated_rewards[state][action]
        noise = np.random.normal(0, 0.1) 
        
        return base_reward + noise
