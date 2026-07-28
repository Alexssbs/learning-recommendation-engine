# AI Smart Learning Platform - Arquitectura y Estructura del Proyecto

Este documento detalla la estructura de carpetas recomendada para el proyecto y el diseño conceptual de la integración entre Aprendizaje No Supervisado y Aprendizaje por Refuerzo.

## 1. Estructura del Repositorio (Basada en MLOps Serverless)

Para acomodar el backend (FastAPI), los modelos (Clustering y RL) y los pipelines de MLOps (GitHub Actions + Databricks), sugerimos la siguiente estructura de carpetas para tu repositorio:

```text
ai-smart-learning-platform/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Pipeline de Integración Continua (Tests, Linting)
│       └── cd.yml                 # Pipeline de Despliegue en Render
│
├── data/                          # Carpeta ignorada en git, solo para desarrollo local
│   ├── raw/                       # Dataset original (ej. OULAD)
│   └── processed/                 # Datos limpios y listos para entrenar
│
├── src/                           # Código fuente principal de la aplicación
│   ├── api/                       # Backend en FastAPI
│   │   ├── main.py                # Punto de entrada de la API
│   │   ├── routes.py              # Endpoints (ej. /recommend, /feedback)
│   │   └── schemas.py             # Validaciones con Pydantic
│   │
│   ├── ml/                        # Lógica de Machine Learning
│   │   ├── clustering/            # Aprendizaje No Supervisado
│   │   │   ├── train_kmeans.py
│   │   │   └── infer_profile.py
│   │   │
│   │   └── reinforcement/         # Aprendizaje por Refuerzo
│   │       ├── environment.py     # Definición de Estados, Acciones y Recompensas
│   │       ├── agent.py           # Algoritmo Q-Learning (Tabla Q)
│   │       └── train_agent.py
│   │
│   └── core/                      # Configuraciones generales
│       └── config.py              # Variables de entorno y rutas
│
├── notebooks/                     # Exploración de datos y pruebas (EDA)
│   └── 01_data_exploration.ipynb
│
├── databricks/                    # Scripts de entrenamiento continuo (CT)
│   ├── job_retrain_models.py      # Script que se ejecuta en Databricks para reentrenar todo
│   └── requirements_db.txt        # Dependencias específicas de Databricks
│
├── tests/                         # Pruebas unitarias para CI
│   ├── test_api.py
│   └── test_ml_models.py
│
├── Dockerfile                     # Para el despliegue en Render
├── requirements.txt               # Dependencias de producción (FastAPI, scikit-learn, numpy)
└── README.md                      # Documentación para el Informe Técnico
```

---

## 2. Diseño Conceptual del Q-Learning (Aprendizaje por Refuerzo)

Para que el desarrollo sea factible antes de la entrega, usaremos **Q-Learning Clásico (Tabular)**. No usaremos redes neuronales profundas (DQN) a menos que sobre tiempo, ya que Q-Learning es más fácil de depurar e implementar desde cero.

### Componentes del Agente RL:

**A. El Estado (State - $S_t$):**
El estado representa la situación actual del estudiante. Lo definimos combinando el perfil descubierto por el Clustering y el nivel actual de riesgo.
*   *Variable 1:* Perfil (Ej: 1=Constante, 2=Nocturno, 3=Irregular, 4=Intensivo)
*   *Variable 2:* Riesgo de abandono reciente (Ej: 0=Bajo, 1=Alto)
*   *Total de Estados posibles:* 4 x 2 = **8 Estados**. (¡Muy pequeño y fácil de manejar!)

**B. Las Acciones (Action - $A_t$):**
Lo que el sistema puede recomendar.
*   Acción 0: Recomendar un **Video Resumen**.
*   Acción 1: Recomendar un **Cuestionario Práctico**.
*   Acción 2: Recomendar **Descanso**.
*   Acción 3: Enviar un **Mensaje Motivacional / Alerta**.
*   *Total de Acciones posibles:* **4 Acciones**.

**C. La Tabla Q (Q-Table):**
Como tenemos 8 estados y 4 acciones, el cerebro de nuestra IA será simplemente una matriz de 8 filas por 4 columnas (8x4). Cada celda almacenará un número que representa "qué tan buena es la acción $A$ en el estado $S$".

**D. La Recompensa (Reward - $R_t$):**
El *feedback* que recibe el sistema del Frontend.
*   **+1**: Si el estudiante hace clic en la recomendación y mejora su tiempo de estudio.
*   **0**: Si el estudiante ignora la recomendación.
*   **-1**: Si el estudiante abandona la plataforma poco después.

---

## 3. Flujo de Trabajo (El "Loop" de Producción)

1. **Inferencia (FastAPI):** El Frontend envía los datos de un estudiante a la API (`/recommend`).
2. **Clustering:** El modelo `infer_profile.py` toma los datos y asigna un Perfil (1 al 4).
3. **Q-Learning:** El `agent.py` toma el Estado (Perfil + Riesgo) busca en la Matriz (Q-Table) la acción con el valor más alto y devuelve la Recomendación.
4. **Feedback:** El usuario interactúa. El Frontend envía el resultado (`/feedback`) a la base de datos.
5. **Re-entrenamiento (Databricks):** Cada noche (o cada semana), Databricks lee los datos de feedback, actualiza los valores de la matriz Q y entrena el Clustering con los nuevos datos históricos. Luego, guarda la nueva Q-Table en MLflow y FastAPI la descarga para usarla al día siguiente.
