import os
import sys
import base64
import time
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from gemini_scanner import GeminiRoundRobinEngine

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

_gemini_engine = None

def get_gemini_engine():
    global _gemini_engine
    if _gemini_engine is None:
        _gemini_engine = GeminiRoundRobinEngine()
    return _gemini_engine

@app.route('/', methods=['GET'])
@app.route('/api', methods=['GET'])
@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    engine = get_gemini_engine()
    return jsonify({
        "status": "SUCCESS",
        "service": "Metal Stencil OCR API",
        "active_keys": len(engine.get_keys())
    })

@app.route('/ocr', methods=['POST'])
@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    img_bgr = None

    # 1. Multipart Form Upload ('file' or 'image')
    file = request.files.get('file') or request.files.get('image')
    if file:
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. JSON Base64 Payload
    elif request.is_json:
        data = request.get_json() or {}
        img_data = data.get('image', '') or data.get('file', '')
        if img_data:
            if ',' in img_data:
                img_data = img_data.split(',', 1)[1]
            try:
                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({
                    "status": "FAILED",
                    "PARTno": None,
                    "error": f"Invalid base64 image: {str(e)}"
                }), 400

    if img_bgr is None:
        return jsonify({
            "status": "FAILED",
            "PARTno": None,
            "error": "No valid image provided via form-data ('file') or JSON ('image')"
        }), 400

    try:
        engine = get_gemini_engine()
        result = engine.scan_part_number(img_bgr)
        status_code = 200 if result.get("status") == "SUCCESS" else 502
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            "status": "FAILED",
            "PARTno": None,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
