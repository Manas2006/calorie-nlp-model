"""Gunicorn configuration file."""
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes - use fewer workers to save memory
workers = 1  # Single worker for free tier
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 100
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "calorie-predictor-api"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Memory optimization
max_requests = 1000
max_requests_jitter = 50 