
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from api.features_and_model import train_select_and_save, load_artifact, predict_next_month_global

class TestModelWorkflow(unittest.TestCase):
    def test_train_and_artifact(self):
        model_path, maes, best = train_select_and_save()
        self.assertTrue(os.path.exists(model_path))
        art = load_artifact(model_path)
        self.assertIn("model", art)

    def test_predict(self):
        art = load_artifact()
        pred = predict_next_month_global(art)
        self.assertIsInstance(pred, float)

if __name__ == "__main__":
    unittest.main()
