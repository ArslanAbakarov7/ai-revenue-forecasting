
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from api.data_ingest import ingest_all_jsons
from api.features_and_model import build_monthly_features

class TestIngestAndFeatures(unittest.TestCase):
    def test_ingest_writes_csv(self):
        out = ingest_all_jsons()
        self.assertTrue(os.path.exists(out))

    def test_features_writes_csv(self):
        f = build_monthly_features()
        self.assertTrue(os.path.exists(f))

if __name__ == "__main__":
    unittest.main()
