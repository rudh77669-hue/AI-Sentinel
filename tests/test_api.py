import unittest
from app import app
from src.database import clear_all_alerts, get_all_alerts
from src.alert_manager import create_alert


class ApiEndpointTests(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_status_endpoint(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["project"], "AI Sentinel")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["feature_count"], 78)

    def test_alert_stats_endpoint(self):
        response = self.client.get("/api/alert-stats")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("total", data)
        self.assertIn("critical", data)
        self.assertIn("unresolved", data)

    def test_sample_threats_endpoint(self):
        response = self.client.get("/api/sample-threats")
        self.assertEqual(response.status_code, 200)
        threats = response.get_json()
        self.assertIsInstance(threats, list)
        self.assertTrue(any(t["name"] == "DDoS" for t in threats))

    def test_simulate_threat_endpoint(self):
        response = self.client.post("/api/simulate-threat", json={"threat_name": "DDoS"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["prediction"]["is_attack"])
        self.assertTrue(data["alert_created"])
        self.assertIsNotNone(data["alert_id"])

    def test_update_alert_endpoint(self):
        # Create alert first
        alert_id = create_alert(
            predicted_label="PortScan",
            severity="High",
            source_ip="198.51.100.99",
            destination_ip="10.0.0.1"
        )
        response = self.client.post(
            f"/api/alerts/{alert_id}",
            json={
                "status": "Investigating",
                "analyst_notes": "Blocked IP at firewall."
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["alert"]["status"], "Investigating")
        self.assertEqual(data["alert"]["analyst_notes"], "Blocked IP at firewall.")


if __name__ == "__main__":
    unittest.main()
