import os
import sys
import base64
import time
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add root directory to sys.path so gemini_scanner can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from gemini_scanner import GeminiRoundRobinEngine, IMPLEMENT_CATALOG, STATE_CODES

app = Flask(__name__)
# Enable CORS for all routes and origins so any web/mobile frontend dev can call it without CORS errors
CORS(app, resources={r"/*": {"origins": "*"}})

# Lazy initialization of Gemini Vision engine
_gemini_engine = None

def get_gemini_engine():
    global _gemini_engine
    if _gemini_engine is None:
        _gemini_engine = GeminiRoundRobinEngine()
    return _gemini_engine

def mat_to_base64_png(mat):
    if mat is None:
        return None
    success, buffer = cv2.imencode('.png', mat)
    if not success:
        return None
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/', methods=['GET'])
@app.route('/api', methods=['GET'])
@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    engine = get_gemini_engine()
    return jsonify({
        "status": "healthy",
        "service": "Laser Cut Metal Stencil OCR API",
        "ai_engine": "Google Gemini Vision AI",
        "round_robin_slots": len(engine.get_keys()),
        "supported_models": list(IMPLEMENT_CATALOG.keys()),
        "version": "2.0.0"
    })

@app.route('/catalog', methods=['GET'])
@app.route('/api/catalog', methods=['GET'])
def catalog():
    return jsonify({
        "status": "success",
        "catalog": IMPLEMENT_CATALOG,
        "states": STATE_CODES
    })

@app.route('/ocr', methods=['POST'])
@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    start_time = time.time()
    img_bgr = None

    # Option A: Multipart file upload ('file' or 'image')
    file = request.files.get('file') or request.files.get('image')
    if file:
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Option B: JSON payload with base64 image ({ "image": "data:image/jpeg;base64,..." or raw base64 })
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
                    "status": "error",
                    "error": f"Invalid base64 image data: {str(e)}"
                }), 400

    if img_bgr is None:
        return jsonify({
            "status": "error",
            "error": "No valid image provided. Send image via multipart form-data (key 'file' or 'image') or JSON body { 'image': '<base64>' }"
        }), 400

    try:
        include_images = request.args.get('include_images', 'true').lower() in ('true', '1', 'yes')
        if request.is_json:
            data = request.get_json() or {}
            if 'include_images' in data:
                include_images = bool(data['include_images'])

        engine = get_gemini_engine()
        result = engine.process_image(img_bgr)

        if not result.get("success"):
            return jsonify({
                "status": "error",
                "error": result.get("error", "Failed to process image with Gemini Vision AI"),
                "elapsed_ms": round((time.time() - start_time) * 1000, 1)
            }), 502

        elapsed_ms = round((time.time() - start_time) * 1000, 1)
        parsed_info = result.get("parsed_info", {})

        response_data = {
            "status": "success",
            "final_code": result.get("final_code", ""),
            "parsed_info": parsed_info,
            "key_used_slot": result.get("key_used_slot"),
            "model_used": result.get("model_used"),
            "best_angle": result.get("best_angle", "0deg"),
            "is_mirrored": result.get("is_mirrored", False),
            "elapsed_ms": elapsed_ms
        }

        if include_images:
            digital_stamp_b64 = mat_to_base64_png(result.get("digital_stamp"))
            cropped_bin_b64 = mat_to_base64_png(result.get("cropped_binary"))

            response_data["digital_stamp_base64"] = f"data:image/png;base64,{digital_stamp_b64}" if digital_stamp_b64 else None
            response_data["cropped_binary_base64"] = f"data:image/png;base64,{cropped_bin_b64}" if cropped_bin_b64 else None

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "elapsed_ms": round((time.time() - start_time) * 1000, 1)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
