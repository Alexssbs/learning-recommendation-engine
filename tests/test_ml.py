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
    
    def test_infer_profile_intensive(self):
        # Un estudiante con muchas horas debería ser clasificado como Intensivo (3)
        profile_id, name = infer_student_profile(study_hours=25, last_score=90)
        self.assertEqual(profile_id, 3)
        self.assertIn("Intensivo", name)
        
    def test_infer_profile_risk(self):
        # Un estudiante con baja nota debería entrar en Riesgo de Abandono (2)
        profile_id, name = infer_student_profile(study_hours=10, last_score=30)
        self.assertEqual(profile_id, 2)
        self.assertIn("Riesgo", name)

if __name__ == "__main__":
    unittest.main()
