def infer_student_profile(study_hours: float, last_score: float) -> tuple[int, str]:
    """
    Simulación del modelo de Clustering (Aprendizaje No Supervisado).
    En un entorno real, aquí cargaríamos un modelo K-Means o HDBSCAN (ej. usando scikit-learn)
    y ejecutaríamos `model.predict([[study_hours, last_score]])`.
    
    Mapeo de Perfiles:
    0: Constante
    1: Nocturno / Irregular
    2: Riesgo de Abandono
    3: Intensivo
    """
    
    if last_score < 50:
        return 2, "Riesgo de Abandono"
    elif study_hours > 20:
        return 3, "Estudiante Intensivo"
    elif study_hours < 10 and last_score >= 50:
        return 1, "Estudio Irregular"
    else:
        return 0, "Estudiante Constante"
