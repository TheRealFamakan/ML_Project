# Utiliser une image Python officielle légère
FROM python:3.9-slim

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires  pour OpenCV et autres bibliothèques
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .

# Création du dossier uploads avec permissions
RUN mkdir -p uploads && chmod 777 uploads

# Lancer l'app avec Gunicorn sur le port dynamique ($PORT) ou 7860 par défaut
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-7860} app:app"]