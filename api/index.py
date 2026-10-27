import os
import sys
import base64
import time
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add root directory to sys.path so ocr_engine can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ocr_engine import StencilOCREngine, IMPLEMENT_CATALOG, STATE_CODES
except ImportError:
    # If in root directory
    from .ocr_engine import StencilOCREngine, IMPLEMENT_CATALOG, STATE_CODES

app = Flask(__name__)
CORS(app)

# Lazy initialization of OCR engine
_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = StencilOCREngine()
    return _engine_instance

def mat_to_base64_png(mat):
    if mat is None:
        return None
    success, buffer = cv2.imencode('.png', mat)
    if not success:
        return None
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/', methods=['GET'])
@app.route('/api', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Laser Cut Metal Stencil OCR API",
        "supported_models": list(IMPLEMENT_CATALOG.keys()),
        "version": "1.0.0"
    })

@app.route('/api/catalog', methods=['GET'])
def catalog():
    return jsonify({
        "catalog": IMPLEMENT_CATALOG,
        "states": STATE_CODES
    })

@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    start_time = time.time()
    img_bgr = None

    # Check 1: multipart file upload
    if 'file' in request.files:
        file = request.files['file']
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check 2: JSON payload with base64 image
    elif request.is_json:
        data = request.get_json() or {}
        img_data = data.get('image', '')
        if img_data:
            if ',' in img_data:
                img_data = img_data.split(',', 1)[1]
            try:
                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400

    if img_bgr is None:
        return jsonify({"error": "No valid image provided via multipart file or JSON base64"}), 400

    try:
        engine = get_engine()
        auto_rotate = request.args.get('auto_rotate', 'true').lower() in ('true', '1', 'yes')
        
        # If client provided json flag
        if request.is_json:
            data = request.get_json() or {}
            if 'auto_rotate' in data:
                auto_rotate = bool(data['auto_rotate'])

        result = engine.process_image(img_bgr, auto_rotate=auto_rotate)
        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        # Convert image outputs to base64 for JSON transport
        cropped_bin_b64 = mat_to_base64_png(result.get("cropped_binary"))
        digital_stamp_b64 = mat_to_base64_png(result.get("digital_stamp"))
        annotated_b64 = mat_to_base64_png(result.get("annotated_image"))

        return jsonify({
            "status": "success",
            "final_code": result.get("final_code", ""),
            "parsed_info": result.get("parsed_info", {}),
            "best_angle": result.get("best_angle", 0),
            "is_mirrored": result.get("is_mirrored", False),
            "elapsed_ms": elapsed_ms,
            "raw_detections": result.get("raw_detections", []),
            "cropped_binary_base64": f"data:image/png;base64,{cropped_bin_b64}" if cropped_bin_b64 else None,
            "digital_stamp_base64": f"data:image/png;base64,{digital_stamp_b64}" if digital_stamp_b64 else None,
            "annotated_image_base64": f"data:image/png;base64,{annotated_b64}" if annotated_b64 else None
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "elapsed_ms": round((time.time() - start_time) * 1000, 1)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
