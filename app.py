from flask import Flask, render_template, request
import os
import cv2  # OpenCV for preprocessing
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = load_model('skin_model.h5')

# Class names from HAM10000
classes = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']

# Symptom matching dict (add more based on research)
disease_symptoms = {
    'Actinic Keratoses': ['rough patches', 'itching'],
    'Basal Cell Carcinoma': ['pearly bump', 'bleeding'],
    'Benign Keratosis': ['waxy spots', 'itching'],
    'Dermatofibroma': ['firm bump', 'tenderness'],
    'Melanoma': ['asymmetrical mole', 'color changes'],
    'Melanocytic Nevi': ['even color', 'no symptoms'],
    'Vascular Lesions': ['red spots', 'swelling']
}

# Recommendations dict (customize; not medical advice)
recommendations_dict = {
    'Actinic Keratoses': 'Use sunscreen, avoid sun. See dermatologist if persists.',
    'Basal Cell Carcinoma': 'Seek medical attention immediately. Biopsy may be needed.',
    'Benign Keratosis': 'Usually harmless; monitor for changes.',
    'Dermatofibroma': 'No treatment needed unless painful.',
    'Melanoma': 'Urgent: See doctor for removal. Early detection saves lives.',
    'Melanocytic Nevi': 'Benign mole; check for ABCDE signs of cancer.',
    'Vascular Lesions': 'May need laser treatment; consult specialist.'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image and symptoms
    file = request.files['image']
    symptoms_input = request.form['symptoms'].lower().split(',')

    # Save image
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Preprocess with OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Normalize color
    img = np.array(img) / 255.0  # Normalize values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict disease
    prediction = model.predict(img)[0]
    disease_idx = np.argmax(prediction)
    disease = classes[disease_idx]
    confidence = prediction[disease_idx] * 100

    # Symptom matching (boost accuracy if match)
    user_symptoms = [s.strip() for s in symptoms_input]
    expected_symptoms = disease_symptoms.get(disease, [])
    match_count = sum(1 for s in user_symptoms if any(s in es.lower() for es in expected_symptoms))
    if match_count > 0:
        confidence += 10  # Simple boost

    # Severity (simplified: based on confidence)
    if confidence > 80:
        severity = 'Mild'
    elif confidence > 50:
        severity = 'Moderate'
    else:
        severity = 'Severe'

    # Recommendations
    recommendations = recommendations_dict.get(disease, 'Consult a doctor for advice.')

    return render_template('results.html', image_path='/'+image_path, disease=disease, severity=severity, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)