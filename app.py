from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and preprocessing components
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from request
        age = data.get('age')
        gender = data.get('gender')
        season = data.get('season')
        destination_type = data.get('destinationType')

        # Prepare input features as DataFrame
        features = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Season': season,
            'Destination Type': destination_type
        }])

        # Apply label encoding
        features['Gender'] = label_encoder['Gender'].transform(features['Gender'])
        features['Season'] = label_encoder['Season'].transform(features['Season'])
        features['Destination Type'] = label_encoder['Destination Type'].transform(features['Destination Type'])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Decode the prediction
        recommended_location = label_encoder['Suggestion'].inverse_transform([prediction])[0]

        return jsonify({'recommendation': recommended_location})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
