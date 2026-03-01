FROM python:3.12-slim

WORKDIR /app

# Install only webapp dependencies (no torch/ML libs)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project (scorers + data are needed at runtime)
COPY . /app

WORKDIR /app/webapp

# Collect static files at build time
RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD gunicorn gamepulse_project.wsgi --bind 0.0.0.0:${PORT:-8000} --workers 2
