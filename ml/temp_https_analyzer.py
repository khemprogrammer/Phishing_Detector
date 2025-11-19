import pandas as pd
import sys
sys.path.append('.')
from ml.features import extract_url_features

file_path = "ml/data/urlset.csv"

def analyze_https_distribution(file_path):
    try:
        df = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    https_values = []
    for i, url in enumerate(df["domain"]):
        if i < 5: # Print first 5 URLs and their has_https value
            features = extract_url_features(url)
            https_values.append(features[8])
            print(f"URL: {url}, has_https: {features[8]}")
        else:
            features = extract_url_features(url)
            https_values.append(features[8])

    https_series = pd.Series(https_values)
    print("Distribution of has_https feature:")
    print(https_series.value_counts())
    print("\nPercentage distribution:")
    print(https_series.value_counts(normalize=True) * 100)

if __name__ == "__main__":
    analyze_https_distribution(file_path)