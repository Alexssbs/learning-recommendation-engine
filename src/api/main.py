from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import sys

# Añadir la carpeta src al PYTHONPATH para poder importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.clustering.infer_profile import infer_student_profile
from ml.reinforcement.agent import QLearningAgent

app = FastAPI(title="AI Smart Learning Platform")

# Inicializamos el Agente RL y cargamos el cerebro matemático
agent = QLearningAgent(num_states=4, num_actions=4)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "q_table.json")
agent.load_model(MODEL_PATH)

# Diccionario para traducir el ID de acción a texto
ACTION_MAP = {
    0: "Ver Video Resumen",
    1: "Hacer Cuestionario Práctico",
    2: "Tomar un Descanso Corto",
    3: "Alerta y Mensaje Motivacional"
}

class StudentData(BaseModel):
    study_hours: float
    last_score: float

# Configurar rutas absolutas para templates y estáticos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Montar archivos estáticos (CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Configurar Jinja2 para las plantillas HTML
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Ruta principal que sirve la interfaz web (Frontend).
    """
    return templates.TemplateResponse(
        request=request, name="index.html", context={"title": "AI Smart Learning Platform"}
    )

@app.post("/api/recommend")
async def get_recommendation(student: StudentData):
    """
    Endpoint (Backend) conectado a la IA (Clustering + Q-Learning).
    """
    # 1. Clustering (Aprendizaje No Supervisado)
    # El sistema descubre a qué grupo pertenece el estudiante sin usar etiquetas.
    profile_id, profile_name = infer_student_profile(student.study_hours, student.last_score)
    
    # 2. Q-Learning (Aprendizaje por Refuerzo)
    # Usamos la política aprendida (Ecuación de Bellman) para tomar la mejor decisión
    # de acuerdo al perfil del estudiante. (Desactivamos la exploración con epsilon=0 para inferencia)
    agent.epsilon = 0.0  
    action_id = agent.choose_action(state=profile_id)
    
    recommended_action = ACTION_MAP.get(action_id, "Acción Desconocida")
    
    return {
        "status": "success",
        "simulated_profile": f"{profile_name} (Perfil {profile_id})",
        "recommended_action": recommended_action,
        "message": "IA Real: Datos procesados por los algoritmos."
    }
