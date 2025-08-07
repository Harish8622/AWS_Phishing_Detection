import unittest
from src.helpers.feature_engineering import compute_simple_features


class TestComputeSimpleFeatures(unittest.TestCase):
    def test_https_url(self):
        url = "https://www.example.com"
        features = compute_simple_features(url)
        self.assertEqual(features["IsHTTPS"], 1)
        self.assertTrue(0 < features["LetterRatioInURL"] <= 1)
        self.assertTrue(features["URLLength"] > 0)

    def test_http_ip_url(self):
        url = "http://192.168.0.1/login?user=admin&pass=123"
        features = compute_simple_features(url)
        self.assertEqual(features["IsHTTPS"], 0)
        self.assertGreaterEqual(features["NoOfAmpersandInURL"], 1)
        self.assertGreater(features["SpacialCharRatioInURL"], 0)

    def test_empty_url(self):
        url = ""
        features = compute_simple_features(url)
        self.assertEqual(features["URLLength"], 0)
        self.assertEqual(features["DomainLength"], 0)
        self.assertEqual(features["LetterRatioInURL"], 0)
        self.assertEqual(features["SpacialCharRatioInURL"], 0)
        self.assertEqual(features["CharContinuationRate"], 0)


if __name__ == "__main__":
    unittest.main()
