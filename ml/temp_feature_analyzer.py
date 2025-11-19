import sys
sys.path.append('.')
from ml.features import extract_url_features

url = "https://www.facebook.com"
features = extract_url_features(url)

feature_names = [
    "url_len", "host_len", "path_len", "query_len", "num_dots", "num_hyphens",
    "num_digits", "num_params", "has_https", "has_at", "has_ip", "tld_len",
    "num_subdirs", "num_fragments", "tokens_count", "ratio_digits",
    "ratio_special", "starts_www"
]

print(f"Features for {url}:")
for name, value in zip(feature_names, features):
    print(f"  {name}: {value}")