import unittest
## adding back in features with custom logic, to ensure consistency when doing inference
import math
from collections import Counter
from urllib.parse import urlparse


def compute_simple_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.hostname or ""

    # 1. URLLength (remove trailing slashes for consistency)
    url_length = len(url.rstrip('/'))

    # 2. DomainLength
    domain_length = len(domain)

    # 3. IsDomainIP (check if domain is an IP address)
    is_domain_ip = int(domain.replace('.', '').isdigit())

    # 4. NoOfSubDomain
    subdomains = domain.split('.')
    no_of_subdomain = len(subdomains) - 2 if len(subdomains) > 2 else 0

    # 5. NoOfLettersInURL
    no_of_letters = sum(1 for c in url if c.isalpha())

    # 6. LetterRatioInURL
    letter_ratio = no_of_letters / url_length if url_length > 0 else 0

    # 7. NoOfDegitsInURL
    no_of_digits = sum(c.isdigit() for c in url)

    # 8. DegitRatioInURL
    digit_ratio = no_of_digits / url_length if url_length > 0 else 0

    # 9. NoOfEqualsInURL
    no_of_equals = url.count('=')

    # 10. NoOfQMarkInURL
    no_of_qmark = url.count('?')

    # 11. NoOfAmpersandInURL
    no_of_ampersand = url.count('&')

    # 12. NoOfOtherSpecialCharsInURL (excluding =, ?, &)
    other_specials = sum(
        1 for c in url if not c.isalnum() and c not in ['=', '?', '&']
    )

    # 13. SpacialCharRatioInURL
    special_char_ratio = other_specials / url_length if url_length > 0 else 0

    # 14. IsHTTPS
    is_https = 1 if parsed_url.scheme.lower() == 'https' else 0

    # 15. CharContinuationRate (repeated adjacent characters)
    repeated_chars = sum(1 for i in range(1, len(url)) if url[i] == url[i - 1])
    char_continuation_rate = repeated_chars / (len(url) - 1) if len(url) > 1 else 0

    # Return as dictionary
    return {
        'URL': url,
        'URLLength': url_length,
        'DomainLength': domain_length,
        'IsDomainIP': is_domain_ip,
        'NoOfSubDomain': no_of_subdomain,
        'NoOfLettersInURL': no_of_letters,
        'LetterRatioInURL': round(letter_ratio, 3),
        'NoOfDegitsInURL': no_of_digits,
        'DegitRatioInURL': round(digit_ratio, 3),
        'NoOfEqualsInURL': no_of_equals,
        'NoOfQMarkInURL': no_of_qmark,
        'NoOfAmpersandInURL': no_of_ampersand,
        'NoOfOtherSpecialCharsInURL': other_specials,
        'SpacialCharRatioInURL': round(special_char_ratio, 3),
        'IsHTTPS': is_https,
        'CharContinuationRate': round(char_continuation_rate, 6)
    }


class TestFeatureEngineering(unittest.TestCase):
    def test_legitimate_url(self):
        url = "https://www.example.com"
        features = compute_simple_features(url)
        self.assertEqual(features['IsHTTPS'], 1)
        self.assertTrue(10 <= features['URLLength'] <= 100)  # Example range
        self.assertTrue(0 <= features['LetterRatioInURL'] <= 1)
    
    def test_phishing_like_url(self):
        url = "http://192.168.0.1/login?user=admin"
        features = compute_simple_features(url)
        self.assertEqual(features['IsHTTPS'], 0)
        self.assertGreaterEqual(features['DomainLength'], 7)  # IP domain
        self.assertEqual(features['IsDomainIP'], 1)
    
    def test_empty_url(self):
        url = ""
        features = compute_simple_features(url)
        self.assertEqual(features['URLLength'], 0)
        self.assertEqual(features['DomainLength'], 0)

if __name__ == "__main__":
    unittest.main()