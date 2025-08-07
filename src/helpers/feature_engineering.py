from urllib.parse import urlparse


def compute_simple_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.hostname or ""

    # 1. URLLength
    url_length = len(url.rstrip("/"))

    # 2. DomainLength
    domain_length = len(domain)

    # 3. NoOfSubDomain
    subdomains = domain.split(".")
    no_of_subdomain = len(subdomains) - 2 if len(subdomains) > 2 else 0

    # 4. LetterRatioInURL
    no_of_letters = sum(1 for c in url if c.isalpha())
    letter_ratio = no_of_letters / url_length if url_length > 0 else 0

    # 5. NoOfAmpersandInURL
    no_of_ampersand = url.count("&")

    # 6. SpacialCharRatioInURL (excluding alphanumerics and &, =, ?)
    other_specials = sum(1 for c in url if not c.isalnum() and c not in ["=", "?", "&"])
    special_char_ratio = other_specials / url_length if url_length > 0 else 0

    # 7. IsHTTPS
    is_https = 1 if parsed_url.scheme.lower() == "https" else 0

    # 8. CharContinuationRate
    repeated_chars = sum(1 for i in range(1, len(url)) if url[i] == url[i - 1])
    char_continuation_rate = repeated_chars / (len(url) - 1) if len(url) > 1 else 0

    return {
        "URL": url,
        "URLLength": url_length,
        "DomainLength": domain_length,
        "NoOfSubDomain": no_of_subdomain,
        "LetterRatioInURL": round(letter_ratio, 3),
        "NoOfAmpersandInURL": no_of_ampersand,
        "SpacialCharRatioInURL": round(special_char_ratio, 3),
        "IsHTTPS": is_https,
        "CharContinuationRate": round(char_continuation_rate, 6),
    }
