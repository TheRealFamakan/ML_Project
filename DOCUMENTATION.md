# Documentation Technique - DeepPnemonia

Cette documentation fournit une description détaillée de l'architecture, des fonctions et des processus du système DeepPnemonia.

## Table des Matières

1. Architecture Globale
2. Backend Flask
3. Frontend Web
4. Modèle VGG16
5. API Reference
6. Prétraitement des Données
7. Pipeline d'Entraînement
8. Tests et Validation
9. Déploiement
10. Sécurité
11. Monitoring
12. Troubleshooting

## 1. Architecture Globale

### 1.1 Architecture en Couches

```
┌─────────────────────────────────────┐
│   Interface Utilisateur Web         │ (HTML/CSS/JS)
│   - Upload files                    │
│   - Display results                 │
├─────────────────────────────────────┤
│   API REST Flask                    │ (Python)
│   - Route handlers                  │
│   - Request validation              │
├─────────────────────────────────────┤
│   Moteur de Prédiction              │ (Python)
│   - Image preprocessing             │
│   - Model inference                 │
│   - Result formatting               │
├─────────────────────────────────────┤
│   Modèle VGG16                      │ (TensorFlow/Keras)
│   - Feature extraction              │
│   - Classification                  │
└─────────────────────────────────────┘
```

### 1.2 Flux de Données

```
User Upload → File Validation → Preprocessing → Model Prediction → Result Formatting → Display
```

### 1.3 Structure des Dossiers

```
pneumonia_app/
├── app.py                      # Point d'entrée Flask
├── best_vgg16_model.h5         # Modèle entraîné (527 MB)
├── requirements.txt            # Dépendances Python
├── Dockerfile                  # Configuration Docker
├── docker-compose.yml          # Orchestration multi-conteneurs
├── .gitignore                  # Fichiers exclus de Git
├── README.md                   # Documentation utilisateur
├── DOCUMENTATION.md            # Cette documentation
├── templates/
│   └── index.html              # Template Jinja2
├── static/
│   ├── css/
│   │   └── style.css           # Styles CSS
│   └── js/
│       └── script.js           # Logique client
├── uploads/                    # Temporaire (auto-créé)
└── tests/                      # Tests unitaires
    ├── test_model.py
    ├── test_api.py
    └── test_preprocessing.py
```

## 2. Backend Flask

### 2.1 Fichier Principal : app.py

#### Imports et Configuration

```python
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import io
import base64

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best_vgg16_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
```

#### Chargement du Modèle

Le modèle est chargé une seule fois au démarrage de l'application pour optimiser les performances.

```python
print("Chargement du modele...")
model = load_model(MODEL_PATH)
print("Modele charge avec succes")
```

Temps de chargement : ~3 secondes (CPU), ~1 seconde (GPU)

### 2.2 Fonctions Utilitaires

#### allowed_file(filename)

Vérifie si l'extension du fichier est autorisée.

```python
def allowed_file(filename):
    """
    Vérifie si le fichier a une extension autorisée
    
    Args:
        filename (str): Nom du fichier avec extension
    
    Returns:
        bool: True si extension autorisée, False sinon
    
    Exemple:
        >>> allowed_file('xray.jpg')
        True
        >>> allowed_file('document.pdf')
        False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

#### preprocess_image(img)

Prépare l'image pour la prédiction (redimensionnement, normalisation).

```python
def preprocess_image(img):
    """
    Prétraite une image pour la prédiction VGG16
    
    Args:
        img (PIL.Image): Image source en RGB
    
    Returns:
        np.array: Tensor normalisé de shape (1, 224, 224, 3)
    
    Pipeline:
        1. Resize à 224x224 (format VGG16)
        2. Conversion PIL -> numpy array
        3. Normalisation [0, 255] -> [0, 1]
        4. Ajout dimension batch (1, H, W, C)
    
    Exemple:
        >>> img = Image.open('xray.jpg')
        >>> tensor = preprocess_image(img)
        >>> tensor.shape
        (1, 224, 224, 3)
    """
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
```

#### predict_image(img)

Effectue la prédiction sur une image prétraitée.

```python
def predict_image(img):
    """
    Prédit si une radiographie montre une pneumonie
    
    Args:
        img (PIL.Image): Image RGB
    
    Returns:
        dict: Résultat de prédiction avec structure:
            {
                'prediction': str ('NORMAL' ou 'PNEUMONIE'),
                'confidence': float (0-100),
                'probabilities': {
                    'NORMAL': float (0-100),
                    'PNEUMONIE': float (0-100)
                }
            }
    
    Logique de décision:
        - Si probabilité PNEUMONIE > 0.5 : classification PNEUMONIE
        - Sinon : classification NORMAL
    
    Performance:
        - Temps d'inférence : ~2.3s (CPU), ~0.5s (GPU)
    
    Exemple:
        >>> img = Image.open('xray.jpg')
        >>> result = predict_image(img)
        >>> print(result)
        {
            'prediction': 'PNEUMONIE',
            'confidence': 92.3,
            'probabilities': {
                'NORMAL': 7.7,
                'PNEUMONIE': 92.3
            }
        }
    """
    img_array = preprocess_image(img)
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    prob_pneumonia = float(prediction * 100)
    prob_normal = float((1 - prediction) * 100)
    
    result = {
        'prediction': 'PNEUMONIE' if prediction > 0.5 else 'NORMAL',
        'confidence': max(prob_pneumonia, prob_normal),
        'probabilities': {
            'NORMAL': prob_normal,
            'PNEUMONIE': prob_pneumonia
        }
    }
    return result
```

### 2.3 Routes Flask

#### GET / - Page d'accueil

```python
@app.route('/')
def index():
    """
    Affiche l'interface web principale
    
    Returns:
        HTML: Template index.html rendu avec Jinja2
    
    URL: http://localhost:5000/
    Méthode: GET
    """
    return render_template('index.html')
```

#### POST /predict - Endpoint de prédiction

```python
@app.route('/predict', methods=['POST'])
def predict():
    """
    Effectue la prédiction sur des images uploadées
    
    Request:
        Content-Type: multipart/form-data
        Body: files[] (Array<File>)
    
    Response:
        Content-Type: application/json
        Body: {
            "results": [
                {
                    "filename": str,
                    "prediction": str ('NORMAL'/'PNEUMONIE'),
                    "confidence": float,
                    "probabilities": {
                        "NORMAL": float,
                        "PNEUMONIE": float
                    },
                    "image_data": str (base64)
                }
            ]
        }
    
    Codes d'erreur:
        - 400: Aucun fichier fourni / Fichier vide
        - 500: Erreur de prédiction
    
    Pipeline:
        1. Validation présence fichiers
        2. Vérification extensions
        3. Lecture et conversion RGB
        4. Prédiction
        5. Encodage base64 pour affichage
        6. Formatage réponse JSON
    
    Exemple:
        >>> import requests
        >>> files = {'files[]': open('xray.jpg', 'rb')}
        >>> response = requests.post('http://localhost:5000/predict', files=files)
        >>> print(response.json())
    """
    if 'files[]' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'Aucun fichier selectionne'}), 400
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                img = Image.open(file.stream).convert('RGB')
                result = predict_image(img)
                result['filename'] = file.filename
                
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                result['image_data'] = f"data:image/jpeg;base64,{img_str}"
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        else:
            results.append({
                'filename': file.filename,
                'error': 'Format de fichier non autorise'
            })
    
    return jsonify({'results': results})
```

#### GET /health - Endpoint de monitoring

```python
@app.route('/health', methods=['GET'])
def health():
    """
    Vérifie l'état du service
    
    Returns:
        JSON: {
            "status": "ok" | "error",
            "model_loaded": bool
        }
    
    Usage:
        - Monitoring par Kubernetes/Docker
        - Vérification pré-déploiement
        - Health checks automatisés
    
    Exemple:
        >>> curl http://localhost:5000/health
        {"status":"ok","model_loaded":true}
    """
    return jsonify({'status': 'ok', 'model_loaded': model is not None})
```

### 2.4 Lancement du Serveur

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

Configuration:
- debug=True : Recharge automatique + messages détaillés (DEVELOPMENT uniquement)
- host='0.0.0.0' : Écoute toutes interfaces réseau
- port=5000 : Port par défaut

Production: Utiliser Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 3. Frontend Web

### 3.1 Templates HTML (index.html)

Structure de base:

```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detecteur de Pneumonie - Deep Learning</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>Detecteur de Pneumonie par Deep Learning</h1>
            <p class="subtitle">Modele VGG16 - Accuracy: 87.2% | AUC: 94.1%</p>
        </header>

        <main>
            <div class="upload-section">
                <div class="drop-zone" id="dropZone">
                    <input type="file" id="fileInput" multiple accept="image/jpeg,image/png,image/jpg" hidden>
                </div>
                <button id="analyzeBtn" class="btn-analyze" disabled>Analyser les images</button>
            </div>

            <div class="results-section" id="resultsSection" style="display: none;">
                <h2>Resultats d'analyse</h2>
                <div id="resultsContainer"></div>
            </div>

            <div class="loader" id="loader" style="display: none;">
                <div class="spinner"></div>
                <p>Analyse en cours...</p>
            </div>
        </main>

        <footer>
            <p class="warning">Avertissement medical: Cet outil est destine a l'aide au diagnostic uniquement.</p>
        </footer>
    </div>

    <script src="{{ url_for('static', filename='js/script.js') }}"></script>
</body>
</html>
```

### 3.2 Styles CSS (style.css)

Points clés:

```css
/* Gradient background */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Drop zone hover effect */
.drop-zone.drag-over {
    background: #e8ecff;
    border-color: #764ba2;
    transform: scale(1.02);
}

/* Résultats en grille responsive */
#resultsContainer {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 30px;
}

/* Barres de probabilité animées */
.bar-fill {
    transition: width 0.5s ease;
}

/* Spinner de chargement */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
```

### 3.3 JavaScript (script.js)

Variables globales:

```javascript
let selectedFiles = [];

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const loader = document.getElementById('loader');
```

Gestion drag-and-drop:

```javascript
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
});
```

Fonction de prédiction:

```javascript
analyzeBtn.addEventListener('click', async () => {
    loader.style.display = 'block';
    resultsSection.style.display = 'none';
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files[]', file);
    });
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        displayResults(data.results);
    } catch (error) {
        alert('Erreur lors de l\'analyse');
    } finally {
        loader.style.display = 'none';
    }
});
```

Affichage des résultats:

```javascript
function createResultCard(result) {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    const predictionClass = result.prediction === 'NORMAL' ? 'normal' : 'pneumonie';
    
    card.innerHTML = `
        <img src="${result.image_data}" alt="${result.filename}">
        <div class="result-content">
            <p class="result-filename">${result.filename}</p>
            <p class="result-prediction ${predictionClass}">${result.prediction}</p>
            <p class="result-confidence">Confiance: ${result.confidence.toFixed(1)}%</p>
            
            <div class="probability-bar">
                <div class="bar-fill normal" style="width: ${result.probabilities.NORMAL}%">
                    ${result.probabilities.NORMAL.toFixed(1)}%
                </div>
            </div>
        </div>
    `;
    
    return card;
}
```

## 4. Modèle VGG16

### 4.1 Architecture Détaillée

```
Input: (224, 224, 3)
↓
VGG16 Base (frozen):
  Block1: Conv3-64 x2 → MaxPool
  Block2: Conv3-128 x2 → MaxPool
  Block3: Conv3-256 x3 → MaxPool
  Block4: Conv3-512 x3 → MaxPool
  Block5: Conv3-512 x3 → MaxPool
Output: (7, 7, 512)
↓
Classifieur (trainable):
  GlobalAveragePooling2D → (512)
  Dense(512, relu) → (512)
  Dropout(0.5)
  Dense(256, relu) → (256)
  Dropout(0.5)
  Dense(1, sigmoid) → (1)
```

### 4.2 Paramètres

- Total paramètres: 17,338,177
- Paramètres entraînables: 2,623,489
- Paramètres gelés: 14,714,688

### 4.3 Code de Construction

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def create_vgg16_model():
    """
    Crée le modèle VGG16 avec transfer learning
    
    Returns:
        Model: Modèle Keras compilé
    """
    base_vgg16 = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_vgg16.trainable = False
    
    x = base_vgg16.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_vgg16.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model
```

## 5. API Reference

### Endpoints

#### GET /

- Description: Affiche l'interface web
- Méthode: GET
- Authentification: Non requise
- Réponse: HTML

#### POST /predict

- Description: Prédiction sur radiographies
- Méthode: POST
- Content-Type: multipart/form-data
- Body: files[] (Array<File>)
- Réponse: JSON
- Codes: 200 (succès), 400 (erreur)

#### GET /health

- Description: Vérification santé service
- Méthode: GET
- Réponse: JSON {"status": "ok", "model_loaded": true}
- Code: 200

## 6. Prétraitement des Données

Pipeline complet appliqué durant l'entraînement:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)
```

## 7. Pipeline d'Entraînement

Étapes complètes pour réentraîner le modèle:

```python
# 1. Charger les données
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# 2. Calculer class weights
class_weight_dict = {0: 1.94, 1: 0.67}

# 3. Configurer callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_vgg16_model.h5', 
    monitor='val_loss', 
    save_best_only=True
)

# 4. Entraîner
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint]
)
```

## 8. Tests et Validation

### Tests Unitaires

```python
import pytest
from app import allowed_file, preprocess_image
from PIL import Image

def test_allowed_file():
    assert allowed_file('test.jpg') == True
    assert allowed_file('test.png') == True
    assert allowed_file('test.pdf') == False

def test_preprocess_image():
    img = Image.open('test_xray.jpg')
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.min() >= 0 and tensor.max() <= 1
```

### Tests d'Intégration

```python
def test_predict_endpoint():
    with app.test_client() as client:
        with open('test_xray.jpg', 'rb') as f:
            response = client.post('/predict', data={'files[]': f})
        assert response.status_code == 200
        assert 'results' in response.json

def test_health_endpoint():
    with app.test_client() as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json['status'] == 'ok'
```

## 9. Déploiement

### Docker

Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```
docker-compose.yml:

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./uploads:/app/uploads
```

Commandes:

```bash
# Build
docker build -t deeppnemonia .

# Run
docker run -p 5000:5000 deeppnemonia

# Docker Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## 10. Sécurité

Mesures implémentées:

- Validation extensions fichiers (whitelist: jpg, jpeg, png)
- Limite taille upload (16MB max)
- Pas de stockage persistant d'images (suppression immédiate)
- CORS configuré (restriction origines en production)
- Pas de credentials en clair dans le code
- Sanitization des noms de fichiers
- Validation MIME type (en plus de l'extension)

Recommandations production:

```python
# Ajouter rate limiting
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["100 per hour"])

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    pass

# HTTPS obligatoire
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
```

## 11. Monitoring

Métriques à surveiller:

```python
import time
import logging

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Middleware timing
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    logging.info(f"{request.method} {request.path} - {response.status_code} - {duration:.3f}s")
    return response
```

Métriques importantes:
- Temps de réponse /predict (objectif: <5s)
- Utilisation mémoire (objectif: <2GB)
- Utilisation CPU/GPU (objectif: <80%)
- Taux d'erreur 4xx/5xx (objectif: <1%)
- Nombre de requêtes par minute

## 12. Troubleshooting

### Erreur: Model not found

```
FileNotFoundError: [Errno 2] No such file or directory: 'best_vgg16_model.h5'
```

Solution:
```bash
# Vérifier présence du fichier
ls -lh best_vgg16_model.h5

# Si absent, télécharger depuis GitHub Releases
wget https://github.com/TheRealFamakan/ML_Project/releases/download/v1.0/best_vgg16_model.h5
```

### Erreur: Out of memory

```
ResourceExhaustedError: OOM when allocating tensor
```

Solution:
```python
# Réduire batch size
# Dans predict_image(), traiter images une par une
# Ou ajouter garbage collection
import gc
gc.collect()
```

### Erreur: CORS policy

```
Access to XMLHttpRequest blocked by CORS policy
```

Solution:
```python
# Configuration CORS plus permissive (développement uniquement)
CORS(app, resources={r"/*": {"origins": "*"}})

# Production: spécifier origines
CORS(app, resources={r"/*": {"origins": ["https://votredomaine.com"]}})
```

### Performance lente

Causes possibles:
1. CPU uniquement (pas de GPU)
2. Images très grandes (>2000x2000)
3. Modèle non optimisé

Solutions:
```bash
# Utiliser GPU
pip install tensorflow-gpu

# Optimiser modèle (TensorFlow Lite)
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Réduire taille images avant upload (côté client)
# Utiliser compression JPEG quality=85
```

### Erreur: Unable to load model

```
ValueError: Unknown layer: DepthwiseConv2D
```

Solution:
```python
# Charger avec custom_objects si nécessaire
from tensorflow.keras.models import load_model

model = load_model('best_vgg16_model.h5', compile=False)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

## Conclusion

Cette documentation couvre tous les aspects techniques du système DeepPnemonia. Pour toute question supplémentaire, consultez le code source sur GitHub ou contactez les mainteneurs du projet.

### Ressources Complémentaires

- Repository GitHub : https://github.com/TheRealFamakan/ML_Project
- Documentation TensorFlow : https://www.tensorflow.org/api_docs
- Documentation Flask : https://flask.palletsprojects.com/
- Dataset Kaggle : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

### Licence

Projet académique développé à l'ENSA Khouribga (2025-2026).

### Auteurs

- CHAMANI Fatima
- CAMARA Famakan
- Encadrant : Pr. Abdelghani GHAZDALI
