import multiprocessing

# Gunicorn configuration
bind = "0.0.0.0:10000"
workers = 1
worker_class = "sync"
timeout = 600  # 10 minutes - for TensorFlow model predictions
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
preload_app = True
