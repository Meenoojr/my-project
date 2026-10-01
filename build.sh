#!/usr/bin/env bash
# Exit on error
set -o errexit

pip install -r requirements.txt

python manage.py makemigrations
python manage.py migrate

python manage.py loaddata datadump.json

if [ "$DJANGO_SUPERUSER_USERNAME" ]; then 
  python manage.py shell -c "
from django.contrib.auth import get_user_model;
User = get_user_model();
if not User.objects.filter(username='$DJANGO_SUPERUSER_USERNAME').exists():
  User.objects.create_superuser('$DJANGO_SUPERUSER_USERNAME', '$DJANGO_SUPERUSER_EMAIL', '$DJANGO_SUPERUSER_PASSWORD');
  print('Superuser created successfully.');
else:
  print('Superuser already exists.');
"
fi