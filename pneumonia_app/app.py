from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ExifTags
import os
import io
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best_vgg16_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# --- Chargement Sécurisé du Modèle ---
model = None
try:
    print(f"Tentative de chargement du modèle depuis: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        # Désactiver GPU pour éviter les erreurs de mémoire sur l'espace gratuit
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        model = load_model(MODEL_PATH)
        print("✅ MODELE CHARGE AVEC SUCCES")
    else:
        print(f"❌ ERREUR: Le fichier {MODEL_PATH} est introuvable !")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE lors du chargement du modèle: {e}")
    # On laisse l'app démarrer pour voir les logs, mais la prédiction échouera.

# --- Authentification ---
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    # Credential "Sérieux" pour la démo (Multi-comptes)
    valid_users = ["admin", "camara.famakan@admin.com", "chamani.fatima@admin.com"]
    
    if username in valid_users and password == "admin123":
        user_display = "Dr. Camara Famakan" if "camara" in username else "Dr. Chamani Fatima" if "chamani" in username else "Administrateur"
        return jsonify({"success": True, "token": "fake-jwt-token-abcd-1234", "user": user_display})
    else:
        return jsonify({"success": False, "message": "Identifiants incorrects"}), 401

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_exif_data(img):
    exif_data = {
        'date': 'Inconnue',
        'camera': 'Inconnu',
        'resolution': f"{img.width}x{img.height}"
    }
    try:
        exif = img._getexif()
        if exif:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == 'DateTimeOriginal':
                    exif_data['date'] = str(value)
                elif tag_name == 'Model':
                    exif_data['camera'] = str(value)
    except Exception:
        pass
    return exif_data

def predict_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    prob_pneumonia = float(prediction * 100)
    prob_normal = float((1 - prediction) * 100)
    
    # Simuler des coordonnees de heatmap pour l'UI (car pas de Grad-CAM implemente)
    # Dans un vrai cas, on utiliserait le gradient du modele
    simulated_heatmap = []
    if prediction > 0.5:
        # Zones aleatoires pour la demo si Pneumonie
        simulated_heatmap = [
            {'x': 100, 'y': 100, 'r': 50, 'intensity': 0.8},
            {'x': 150, 'y': 120, 'r': 40, 'intensity': 0.6}
        ]

    result = {
        'prediction': 'PNEUMONIE' if prediction > 0.5 else 'NORMAL',
        'confidence': max(prob_pneumonia, prob_normal),
        'probabilities': {
            'NORMAL': prob_normal,
            'PNEUMONIE': prob_pneumonia
        },
        'heatmap_zones': simulated_heatmap
    }
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Erreur Serveur: Le modèle IA n\'est pas chargé. (Voir Logs)'}), 500

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
                result['metadata'] = get_exif_data(img)
                
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)