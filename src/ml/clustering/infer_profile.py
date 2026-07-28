import joblib
import os
import pandas as pd

def infer_student_profile(study_hours: float, last_score: float) -> tuple[int, str]:
    """
    Inferencia de Clustering usando el modelo real K-Means entrenado.
    """
    # Rutas a los modelos guardados
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    # Si los modelos reales no existen (aún no se han entrenado), usamos un fallback simple
    if not os.path.exists(kmeans_path) or not os.path.exists(scaler_path):
        return 0, "Base Profile (K-Means not trained)"
        
    kmeans = joblib.load(kmeans_path)
    scaler = joblib.load(scaler_path)
    
    # Preprocesamiento de la entrada para emparejar con el formato del dataset UCI
    # Convertir horas a la escala de estudio del dataset (1 a 4)
    if study_hours < 2: studytime = 1
    elif study_hours <= 5: studytime = 2
    elif study_hours <= 10: studytime = 3
    else: studytime = 4
    
    # Asumimos promedios para valores no proveídos (inasistencias y fallos)
    absences = 2 
    failures = 0
    
    # Convertir nota sobre 100 a nota sobre 20 (escala europea del dataset)
    g3_score = (last_score / 100) * 20
    
    # Crear un DataFrame con los nombres de características correctos
    df_input = pd.DataFrame([[studytime, absences, failures, g3_score]], 
                            columns=['studytime', 'absences', 'failures', 'G3'])
    
    # Escalar y Predecir
    X_scaled = scaler.transform(df_input)
    cluster_id = int(kmeans.predict(X_scaled)[0])
    
    # Mapeo de nombres descriptivos (puede variar según cómo agrupó el K-Means)
    nombres_perfiles = {
        0: "Group 0: Average Performance",
        1: "Group 1: High Risk",
        2: "Group 2: Academic Excellence",
        3: "Group 3: Irregular Study Habits"
    }
    
    return cluster_id, nombres_perfiles.get(cluster_id, "Unknown")

