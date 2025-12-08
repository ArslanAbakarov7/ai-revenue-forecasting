
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from api.app import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)

    def test_logfile(self):
        r = self.client.get("/logfile")
        self.assertEqual(r.status_code, 200)

if __name__ == "__main__":
    unittest.main()
