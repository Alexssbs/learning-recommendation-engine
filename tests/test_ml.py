import unittest
import sys
import os

# Asegurar que el entorno de pruebas pueda encontrar el módulo src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.clustering.infer_profile import infer_student_profile

class TestMLModels(unittest.TestCase):
    """
    Pruebas unitarias para validar que el modelo matemático no se rompa
    con nuevos cambios. GitHub Actions ejecutará esto automáticamente.
    """
    
    def test_infer_profile_returns_valid_tuple(self):
        # Aseguramos que la inferencia (con el modelo K-Means real) retorne ID numérico y texto en inglés
        profile_id, name = infer_student_profile(study_hours=25, last_score=90)
        
        self.assertIsInstance(profile_id, int)
        self.assertIsInstance(name, str)
        self.assertIn("Group", name) # Debe retornar uno de los grupos en inglés
        
    def test_infer_profile_risk(self):
        # Comprobar que los estudiantes con notas extremas sean procesados sin error
        profile_id, name = infer_student_profile(study_hours=1, last_score=10)
        
        self.assertTrue(0 <= profile_id <= 3) # Debe ser un cluster entre 0 y 3

if __name__ == "__main__":
    unittest.main()
