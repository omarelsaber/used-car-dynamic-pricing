#!/usr/bin/env python
"""Quick test script for Cardekho V5.0 API"""

import requests
import json
import time

# Wait for API to reload
time.sleep(2)

payload = {
    'name': 'Maruti Swift Dzire',
    'year': 2014,
    'km_driven': 145500,
    'fuel': 'Diesel',
    'seller_type': 'Individual',
    'transmission': 'Manual',
    'owner': 'First Owner',
    'mileage': 23.4,
    'engine': 1248,
    'max_power': 74.0,
    'seats': 5
}

print('=' * 60)
print('🚗 CARDEKHO V5.0 API TEST')
print('=' * 60)
print()

try:
    response = requests.post('http://localhost:8001/predict', json=payload, timeout=5)
    data = response.json()
    price = data['predicted_price']
    currency = data['currency']
    model = data['model_version']
    
    print(f'✅ API Status: {response.status_code} OK')
    print(f'🎯 Predicted Price: Rs. {price:,.2f}')
    print(f'💱 Currency: {currency}')
    print(f'📦 Model: {model}')
    print()
    print('✨ SUCCESS! API is ready for frontend.')
    print('=' * 60)
    
except Exception as e:
    print(f'❌ Error: {e}')
    print('API may still be reloading. Please wait a moment.')
