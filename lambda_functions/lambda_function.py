import boto3
import json
from urllib.parse import urlparse

def compute_final_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.hostname or ""

    # 1. URLLength (remove trailing slashes for consistency)
    url_length = len(url.rstrip('/'))

    # 2. DomainLength
    domain_length = len(domain)

    # 3. NoOfSubDomain
    subdomains = domain.split('.')
    no_of_subdomain = len(subdomains) - 2 if len(subdomains) > 2 else 0

    # 4. LetterRatioInURL
    no_of_letters = sum(1 for c in url if c.isalpha())
    letter_ratio = no_of_letters / url_length if url_length > 0 else 0

    # 5. NoOfAmpersandInURL
    no_of_ampersand = url.count('&')

    # 6. SpacialCharRatioInURL (count of non-alphanumeric and not '&')
    other_specials = sum(1 for c in url if not c.isalnum() and c != '&')
    special_char_ratio = other_specials / url_length if url_length > 0 else 0

    # 7. IsHTTPS
    is_https = 1 if parsed_url.scheme.lower() == 'https' else 0

    # 8. CharContinuationRate (repeated adjacent characters)
    repeated_chars = sum(1 for i in range(1, len(url)) if url[i] == url[i - 1])
    char_continuation_rate = repeated_chars / (len(url) - 1) if len(url) > 1 else 0

    return [
        url_length, domain_length, no_of_subdomain, round(letter_ratio, 3),
        no_of_ampersand, round(special_char_ratio, 3),
        is_https, round(char_continuation_rate, 6)
    ]

def lambda_handler(event, context):
    # Parse request body (JSON with 'url')
    body = json.loads(event['body'])
    url = body.get('url', '')

    # Compute final features
    features = compute_final_features(url)
    csv_line = ','.join(map(str, features))

    # Call SageMaker endpoint
    runtime = boto3.client('sagemaker-runtime')
    response = runtime.invoke_endpoint(
        EndpointName='sagemaker-xgboost-2025-05-30-15-30-40-375', 
        ContentType='text/csv',
        Body=csv_line
    )

    # Parse SageMaker response
    result = float(response['Body'].read().decode('utf-8').strip())
    predicted_label = 'legitimate' if result >= 0.5 else 'phishing'

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
            'features': csv_line,
            'prediction_score': result,
            'prediction_label': predicted_label
        })
    }