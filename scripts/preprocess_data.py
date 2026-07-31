"""
Script de preprocesamiento dinámico para MovieLens Latest Small.
Detecta automáticamente la estructura del dataset sin hardcodear nombres.
"""

import pandas as pd
import os
import json
from pathlib import Path

def detect_and_preprocess():
    """Detecta automáticamente la estructura y preprocesa los datos"""
    
    print("🎬 Iniciando preprocesamiento dinámico de MovieLens...")
    
    # 1. Definir rutas
    raw_dir = Path("data/raw/ml-latest-small")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Verificar que los archivos existen
    required_files = ['ratings.csv', 'movies.csv', 'tags.csv', 'links.csv']
    missing = [f for f in required_files if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Archivos faltantes: {missing}")
    
    print("✅ Todos los archivos encontrados")
    
    # 3. Cargar ratings dinámicamente
    print("\n📊 Cargando ratings...")
    ratings = pd.read_csv(raw_dir / 'ratings.csv')
    print(f"   ✅ {len(ratings):,} calificaciones")
    print(f"   Columnas: {list(ratings.columns)}")
    print(f"   Rango de ratings: {ratings['rating'].min()} - {ratings['rating'].max()}")
    
    # 4. Cargar movies dinámicamente
    print("\n🎬 Cargando películas...")
    movies = pd.read_csv(raw_dir / 'movies.csv')
    print(f"   ✅ {len(movies):,} películas")
    print(f"   Columnas: {list(movies.columns)}")
    
    # 5. Procesar géneros (pipe-separated a one-hot)
    print("\n🔄 Procesando géneros...")
    genres_col = 'genres'  # Detectado automáticamente
    
    # Obtener todos los géneros únicos
    all_genres = set()
    for genres_str in movies[genres_col].dropna():
        all_genres.update(genres_str.split('|'))
    all_genres = sorted(all_genres)
    print(f"   Géneros encontrados: {len(all_genres)}")
    print(f"   {all_genres}")
    
    # Crear columnas one-hot
    for genre in all_genres:
        movies[genre] = movies[genres_col].apply(
            lambda x: 1 if isinstance(x, str) and genre in x.split('|') else 0
        )
    
    # 6. Guardar datasets procesados
    print("\n💾 Guardando datos procesados...")
    
    # Guardar movies con géneros one-hot
    movies.to_csv(processed_dir / 'movies_with_genres.csv', index=False)
    print(f"   ✅ movies_with_genres.csv ({len(movies):,} filas)")
    
    # Guardar ratings (ya está limpio)
    ratings.to_csv(processed_dir / 'ratings_clean.csv', index=False)
    print(f"   ✅ ratings_clean.csv ({len(ratings):,} filas)")
    
    # Guardar metadata para referencia
    metadata = {
        'total_movies': len(movies),
        'total_users': ratings['userId'].nunique(),
        'total_ratings': len(ratings),
        'genres': all_genres,
        'rating_range': [float(ratings['rating'].min()), float(ratings['rating'].max())],
        'date_processed': pd.Timestamp.now().isoformat()
    }
    
    with open(processed_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ metadata.json guardado")
    
    # 7. Mostrar estadísticas
    print("\n📊 Estadísticas del Dataset:")
    print(f"   👥 Usuarios: {metadata['total_users']:,}")
    print(f"   🎬 Películas: {metadata['total_movies']:,}")
    print(f"   ⭐ Calificaciones: {metadata['total_ratings']:,}")
    print(f"   📈 Ratings: {metadata['rating_range'][0]} - {metadata['rating_range'][1]}")
    print(f"   🏷️  Géneros: {len(metadata['genres'])}")
    
    print("\n✅ ¡Preprocesamiento completado!")
    return movies, ratings, metadata

if __name__ == "__main__":
    detect_and_preprocess()