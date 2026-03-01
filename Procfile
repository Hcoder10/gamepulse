web: bash -c 'cd webapp && python manage.py collectstatic --noinput && gunicorn gamepulse_project.wsgi --bind 0.0.0.0:$PORT --workers 2'
