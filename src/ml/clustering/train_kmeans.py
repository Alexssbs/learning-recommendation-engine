import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_and_save_clustering():
    print("Iniciando entrenamiento de Clustering con datos REALES...")
    
    # 1. Cargar el dataset real (Estudiantes de la Universidad de Irvine - UCI)
    # Se usa el delimitador ';' ya que es un CSV europeo
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    data_path = os.path.join(raw_dir, "student-mat.csv")
    
    if not os.path.exists(data_path):
        import urllib.request
        import zipfile
        print("Dataset no encontrado localmente. Descargando desde UCI Machine Learning Repository...")
        zip_path = os.path.join(raw_dir, "student.zip")
        urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print("Descarga y extracción completada.")
        
    df = pd.read_csv(data_path, sep=';')
    
    # 2. Selección de Características (Feature Engineering)
    # Elegimos variables que definen el "Perfil de Aprendizaje"
    # studytime: 1(<2 horas), 2(2 a 5 horas), 3(5 a 10 horas), 4(>10 horas)
    # absences: inasistencias
    # failures: número de clases reprobadas anteriormente
    # G3: Nota final (0 a 20)
    features = ['studytime', 'absences', 'failures', 'G3']
    X = df[features]
    
    # 3. Escalar los datos (Muy importante en K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3.5 Guardar datos procesados (Pipeline de Datos)
    # Aquí es donde le damos uso a la carpeta data/processed
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    df_processed = pd.DataFrame(X_scaled, columns=features)
    df_processed.to_csv(os.path.join(processed_dir, "student_processed.csv"), index=False)
    print(f"Datos limpios y procesados guardados en data/processed/student_processed.csv")
    
    # 4. Entrenar el modelo No Supervisado (K-Means)
    # Buscamos 4 perfiles de estudiantes
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    print("\nResultados del Clustering:")
    df['Cluster'] = kmeans.labels_
    for i in range(4):
        cluster_data = df[df['Cluster'] == i]
        print(f"Perfil {i}: {len(cluster_data)} estudiantes. Promedio de nota: {cluster_data['G3'].mean():.2f}")
    
    # 5. Guardar los modelos (Artefactos) e Integrar con Databricks (MLflow)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Intenta registrar en Databricks usando MLflow
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.metrics import silhouette_score
        
        print("\nConectando a Databricks MLflow...")
        mlflow.set_tracking_uri("databricks")
        # Usa el nombre de experimento por defecto o el que viene del entorno (ej. de ct.yml)
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/mlops_clustering_prod")
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Registrar hiperparámetros
            mlflow.log_param("n_clusters", 4)
            mlflow.log_param("random_state", 42)
            
            # Calcular y registrar métricas (qué tan bien se separaron los grupos)
            score = silhouette_score(X_scaled, kmeans.labels_)
            mlflow.log_metric("silhouette_score", score)
            
            # Subir el modelo a la nube de Databricks
            mlflow.sklearn.log_model(kmeans, "kmeans_model")
            print(f"✅ ¡Modelo registrado exitosamente en Databricks! (Silhouette Score: {score:.3f})")
    except Exception as e:
        print(f"\n⚠️ Advertencia: No se pudo conectar a Databricks MLflow ({e}). Guardando solo en local.")

    # Guardar localmente para que la App Web (FastAPI) pueda cargarlos
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    print(f"\nModelos reales guardados localmente en {models_dir}")

if __name__ == "__main__":
    train_and_save_clustering()
