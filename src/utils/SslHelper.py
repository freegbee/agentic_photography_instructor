def create_unverified_ssl_context():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context