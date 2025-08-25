# Gunicorn configuration for stability
import multiprocessing
import os

# Server socket
# Respect Render's PORT environment variable
PORT = os.getenv("PORT", "10000")
bind = f"0.0.0.0:{PORT}"
backlog = 2048

# Worker processes
workers = 1  # Single worker for Render free tier
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True  # Preload app to avoid memory issues

# Timeout settings
timeout = 120  # Increase timeout to 2 minutes
keepalive = 2
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "rag-chatbot"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (not needed for Render)
keyfile = None
certfile = None
