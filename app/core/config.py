"""Web-layer settings only."""

import os

from dotenv import load_dotenv

load_dotenv()

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_V1_PREFIX = "/api/v1"