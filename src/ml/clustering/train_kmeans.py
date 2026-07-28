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
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "raw", "student-mat.csv")
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
    
    # 5. Guardar los modelos (Artefactos)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    print(f"\nModelos reales guardados en {models_dir}")

if __name__ == "__main__":
    train_and_save_clustering()
