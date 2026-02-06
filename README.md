# DeepPnemonia - Système de Détection Automatisée de Pneumonie

Projet de Machine Learning développé dans le cadre du cours ML à l'ENSA Khouribga. Ce système utilise le deep learning (VGG16) pour détecter automatiquement la pneumonie sur des radiographies thoraciques avec une précision de 87.2%.

## Auteurs

- CHAMANI Fatima
- CAMARA Famakan
- Encadrant : Pr. Abdelghani GHAZDALI
- Année universitaire : 2025-2026

## Performances du Modèle

- Accuracy : 87.18%
- Precision : 89.34%
- Recall : 90.26%
- AUC : 94.09%
- Temps de prédiction : 2.3 secondes par image

## Fonctionnalités

- Upload multiple de radiographies (JPEG/PNG, max 16MB)
- Traitement batch (jusqu'à 10 images simultanément)
- Interface web drag-and-drop intuitive
- Visualisation des probabilités en temps réel
- API REST pour intégration dans systèmes hospitaliers
- Déploiement Docker pour portabilité maximale

## Technologies Utilisées

- Backend : Flask, Gunicorn
- Deep Learning : TensorFlow 2.x, Keras
- Frontend : HTML5, CSS3, JavaScript
- Déploiement : Docker, Docker Compose
- Modèle : VGG16 avec Transfer Learning

## Prérequis

- Python >= 3.8
- pip >= 21.0
- (Optionnel) GPU CUDA pour accélération

## Installation Locale

```bash
# Cloner le repository
git clone https://github.com/TheRealFamakan/ML_Project.git
cd ML_Project/pneumonia_app

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python app.py
```

Accédez ensuite à l'interface web : http://localhost:5000

## Installation avec Docker

```bash
# Construire l'image
docker build -t deeppnemonia .

# Lancer le conteneur
docker run -p 5000:5000 deeppnemonia
```

Accédez à l'application : http://localhost:5000

## Utilisation

### Interface Web

1. Ouvrez http://localhost:5000 dans votre navigateur
2. Glissez-déposez vos radiographies ou cliquez sur "Parcourir les fichiers"
3. Cliquez sur "Analyser les images"
4. Consultez les résultats avec probabilités détaillées

### API REST

```bash
# Endpoint de prédiction
curl -X POST http://localhost:5000/predict -F "files[]=@xray1.jpg" -F "files[]=@xray2.jpg"
```

Réponse JSON :
```json
{
  "results": [
    {
      "filename": "xray1.jpg",
      "prediction": "PNEUMONIE",
      "confidence": 92.3,
      "probabilities": {
        "NORMAL": 7.7,
        "PNEUMONIE": 92.3
      },
      "image_data": "data:image/jpeg;base64,..."
    }
  ]
}
```

```bash
# Endpoint de santé
curl http://localhost:5000/health
```

Réponse :
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Structure du Projet

```
pneumonia_app/
├── app.py                      # Backend Flask
├── best_vgg16_model.h5         # Modèle VGG16 entraîné
├── requirements.txt            # Dépendances Python
├── Dockerfile                  # Configuration Docker
├── templates/
│   └── index.html              # Interface web
├── static/
│   ├── css/
│   │   └── style.css           # Styles
│   └── js/
│       └── script.js           # Logique frontend
└── uploads/                    # Dossier temporaire (auto-créé)
```

## Dataset

- Source : Kaggle Chest X-Ray Images (Pneumonia)
- Origine : Guangzhou Women and Children's Medical Center, Chine
- Population : Patients pédiatriques (1-5 ans)
- Composition : 5,856 radiographies (1,575 NORMAL, 4,265 PNEUMONIE)
- Split : 80% train, 20% validation, test séparé (624 images)

## Méthodologie

### Prétraitement
- Redimensionnement : 224x224 pixels
- Normalisation : Division par 255 (conversion [0,1])
- Augmentation : Rotation (±15°), Translation (±10%), Zoom (±15%), Flip horizontal

### Architecture VGG16
- Base : VGG16 pré-entraîné sur ImageNet (poids gelés)
- Classifieur : GlobalAveragePooling2D -> Dense(512, relu) -> Dropout(0.5) -> Dense(256, relu) -> Dropout(0.5) -> Dense(1, sigmoid)
- Optimizer : Adam (learning rate = 0.0001)
- Loss : Binary Crossentropy
- Class Weights : {NORMAL: 1.94, PNEUMONIE: 0.67}

### Entraînement
- Epochs : 20 (early stopping patience = 5)
- Batch size : 32
- Callbacks : EarlyStopping, ModelCheckpoint
- Durée : ~3 heures (CPU), ~45 minutes (GPU)

## Résultats

### Matrice de Confusion (Test Set)

|                | Pred. NORMAL | Pred. PNEUMONIE | Total |
|----------------|--------------|-----------------|-------|
| NORMAL         | 192 (TN)     | 42 (FP)         | 234   |
| PNEUMONIE      | 38 (FN)      | 352 (TP)        | 390   |
| Total          | 230          | 394             | 624   |

### Métriques Cliniques
- PPV (Valeur Prédictive Positive) : 89.3%
- NPV (Valeur Prédictive Négative) : 83.5%
- Specificity : 82.05%
- Sensitivity (Recall) : 90.26%

### Comparaison avec l'État de l'Art

| Étude                    | Modèle        | Accuracy | Recall |
|--------------------------|---------------|----------|--------|
| Kermany et al. (2018)    | Inception-v3  | 92.8%    | 93.2%  |
| Rajpurkar et al. (2017)  | CheXNet       | 87.8%    | 89.5%  |
| Notre étude              | VGG16         | 87.2%    | 90.3%  |
| Radiologues humains      | -             | 85-90%   | 88-92% |

## Déploiement en Production

### Heroku
```bash
heroku create deeppnemonia-app
git push heroku master
heroku open
```

### Serveur dédié (Ubuntu)
```bash
# Installation Docker
sudo apt update
sudo apt install docker.io docker-compose

# Déploiement
git clone https://github.com/TheRealFamakan/ML_Project.git
cd ML_Project/pneumonia_app
docker-compose up -d

# Nginx reverse proxy
sudo nano /etc/nginx/sites-available/deeppnemonia
sudo nginx -t
sudo systemctl restart nginx
```

## Limitations

- Dataset pédiatrique uniquement (généralisation aux adultes non validée)
- 38 faux négatifs (9.7% des pneumonies non détectées)
- Classification binaire (pas de distinction bactérienne/virale)
- Pas d'explicabilité (boîte noire, pas de Grad-CAM)

## Perspectives

### Court terme (3-6 mois)
- Fine-tuning du modèle (dégeler les dernières couches VGG16)
- Ajustement du seuil de décision (0.5 -> 0.3 pour augmenter recall)
- Intégration Grad-CAM pour visualisation des zones pathologiques

### Moyen terme (6-12 mois)
- Extension multi-classes (NORMAL / PNEUMONIE BACTÉRIENNE / VIRALE / COVID-19)
- Ensemblage de modèles (VGG16 + ResNet50 + EfficientNet)
- Support format DICOM (standard hospitalier)

### Long terme (1-2 ans)
- Validation clinique multicentrique (Maroc, France, Sénégal)
- Certification médicale (Marquage CE, FDA clearance)
- Intégration PACS (Picture Archiving and Communication System)

## Licence

Ce projet est développé à des fins académiques dans le cadre du cours de Machine Learning à l'ENSA Khouribga.

## Avertissement Médical

Cet outil est destiné à l'aide au diagnostic uniquement. Une confirmation par un professionnel de santé qualifié est obligatoire. Le modèle ne doit pas être utilisé comme unique source de décision médicale.

## Contact

- GitHub : https://github.com/TheRealFamakan/ML_Project
- Email : [Votre email]

## Remerciements

- Pr. Abdelghani GHAZDALI pour son encadrement
- ENSA Khouribga pour les ressources informatiques
- Communauté Kaggle pour le dataset
- Contributeurs TensorFlow/Keras

## Références

1. Kermany, D.S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172(5), 1122-1131.
2. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv:1711.05225.
3. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR 2015.
