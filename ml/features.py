import re
from urllib.parse import urlparse, parse_qs
import numpy as np
import tldextract

suspicious_tokens = [
    "secure","update","verify","bank","confirm","wallet",
    "password","ebayisapi","webscr","paypal"
]

def is_ip(host):
    return re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host) is not None

def extract_url_features(url):
    original_url = url
    if not isinstance(url, str):
        url = ""
    if not re.match(r"^[a-zA-Z]+://", url):
        url = "http://" + url
    try:
        p = urlparse(url)
        ext = tldextract.extract(url if isinstance(url, str) else "")
        host = p.hostname or ""
        path = p.path or ""
        query = p.query or ""
        url_len = len(url)
        host_len = len(host)
        path_len = len(path)
        query_len = len(query)
        num_dots = url.count(".")
        num_hyphens = url.count("-")
        num_digits = sum(c.isdigit() for c in url)
        num_params = len(parse_qs(query))
        has_https = 1 if original_url.lower().startswith("https://") else 0
        has_at = 1 if "@" in url else 0
        has_ip = 1 if is_ip(host) else 0
        tld_len = len(ext.suffix or "")
        num_subdirs = path.strip("/").count("/")
        num_fragments = url.count("#")
        tokens_count = sum(1 for t in suspicious_tokens if t in url.lower())
        ratio_digits = num_digits / url_len if url_len else 0.0
        special_chars = sum(c in "!@#$%^\u0026*()_+-={}[]|\\:;\"'\u003c\u003e,.?/" for c in url)
        ratio_special = special_chars / url_len if url_len else 0.0
        starts_www = 1 if host.startswith("www") else 0
        return [
            url_len,host_len,path_len,query_len,num_dots,num_hyphens,num_digits,
            num_params,has_https,has_at,has_ip,tld_len,num_subdirs,num_fragments,
            tokens_count,ratio_digits,ratio_special,starts_www
        ]
    except ValueError:
        # Return a default set of features (e.g., all zeros) for malformed URLs
        return [0] * 18

class FeatureExtractor:
    def transform(self, urls):
        return np.array([extract_url_features(u) for u in urls], dtype=np.float32)