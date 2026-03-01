import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

# Add the project root to sys.path so we can import scorers/src/pipeline
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "gamepulse-insecure-dev-key")
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
CSRF_TRUSTED_ORIGINS = [
    o.strip() for o in os.getenv("CSRF_TRUSTED_ORIGINS", "").split(",") if o.strip()
] + [
    "https://web-production-fe47d.up.railway.app",
    "https://*.up.railway.app",
]

INSTALLED_APPS = [
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "core",
    "copilot",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_AGE = 60 * 60 * 24 * 30  # 30 days

ROOT_URLCONF = "gamepulse_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "gamepulse_project.wsgi.application"
DATABASES = {}

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Model Inference Endpoints (for Luau Copilot)
MODEL_ENDPOINT_URL = os.getenv("MODEL_ENDPOINT_URL", "")
SFT_ENDPOINT_URL = os.getenv("SFT_ENDPOINT_URL", "")
RFT_ENDPOINT_URL = os.getenv("RFT_ENDPOINT_URL", "")

# Results data paths (for pipeline dashboard)
RESULTS_DIR = PROJECT_ROOT / "results"
