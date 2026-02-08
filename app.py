#!/usr/bin/env python3
"""
Dzud Risk API - Flask Web Service
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from risk_predictor import DzudRiskPredictor
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize predictor
predictor = DzudRiskPredictor()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict dzud risk
    
    Request body:
    {
        "lat": 43.5,
        "lon": 104.4,
        "livestock": {
            "sheep": 200,
            "goat": 150,
            "cattle": 50,
            "horse": 30,
            "camel": 10
        },
        "month": 1  (optional, defaults to current month)
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if 'lat' not in data or 'lon' not in data:
            return jsonify({'error': 'lat and lon are required'}), 400
        
        if 'livestock' not in data:
            return jsonify({'error': 'livestock data is required'}), 400
        
        lat = float(data['lat'])
        lon = float(data['lon'])
        livestock = data['livestock']
        month = data.get('month', None)
        
        # Predict
        result = predictor.predict(lat, lon, livestock, month)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get available locations"""
    locations = predictor.weather_data[['aimag', 'soum', 'lat', 'lon']].drop_duplicates()
    result = locations.to_dict('records')
    print(f"📍 Returning {len(result)} locations")
    for loc in result:
        print(f"   - {loc['soum']}")
    return jsonify(result)

@app.route('/api/risk-map', methods=['POST'])
def get_risk_map():
    """
    Get risk levels for all locations
    
    Request body:
    {
        "livestock": {...},
        "month": 1
    }
    """
    try:
        data = request.get_json()
        livestock = data.get('livestock', {})
        month = data.get('month', None)
        
        # Get all unique locations
        locations = predictor.weather_data[['aimag', 'soum', 'lat', 'lon']].drop_duplicates()
        
        # Calculate risk for each location
        risk_map = []
        for _, loc in locations.iterrows():
            result = predictor.predict(loc['lat'], loc['lon'], livestock, month)
            risk_map.append({
                'soum': loc['soum'],
                'lat': loc['lat'],
                'lon': loc['lon'],
                'risk_score': result['risk']['score'],
                'risk_level': result['risk']['level'],
                'risk_label': result['risk']['label'],
                'color': result['risk']['color']
            })
        
        return jsonify(risk_map)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/geojson', methods=['GET'])
def get_geojson():
    """Get Omnogovi soum boundaries from GeoJSON"""
    try:
        import json
        
        # Read simplified Omnogovi GeoJSON
        with open('omnogovi_soums_simple.geojson', 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        return jsonify(geojson_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor.has_model,
        'weather_data_rows': len(predictor.weather_data)
    })

if __name__ == '__main__':
    print("🚀 Starting Dzud Risk API...")
    print("📍 API endpoint: http://localhost:5001/api/predict")
    print("🌐 Web interface: http://localhost:5001/")
    app.run(debug=True, host='0.0.0.0', port=5001)
