"""
Gemini Vision Round-Robin OCR Engine for Laser Cut Metal Stencils
Uses Google Gemini Vision with 4-key round-robin rotation, automatic error failover,
and industrial implement stencil parsing.
"""

import os
import json
import time
import base64
import urllib.request
import urllib.error
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List

# Centralized Implement Catalog based on manufacturer specifications
IMPLEMENT_CATALOG = {
    "444": {"name": "REG SMART / REG", "series": "Regular Series", "sizes": ["5ft", "5.5ft", "6.5ft"]},
    "445": {"name": "REG V1 / SEMI MSS", "series": "Regular / Semi Series", "sizes": ["5ft", "5.5ft", "1.55m"]},
    "446": {"name": "TUSKER / SEMI HSS / SEMI CHA", "series": "Heavy Duty / Semi Series", "sizes": ["5ft", "6ft", "1.65m"]},
    "447": {"name": "VICTOR / 8ft SPM", "series": "Special Purpose Heavy Series", "sizes": ["7ft", "8ft", "2.0m"]},
    "3528": {"name": "REGULAR IMPLEMENT", "series": "Standard Centralized Implement", "sizes": ["Various"]},
    "3674": {"name": "BRACKET / STRUCTURAL MOUNT", "series": "Implement Bracket Sub-Assembly", "sizes": ["Standard"]},
}

STATE_CODES = {
    "GJ": "Gujarat",
    "MH": "Maharashtra",
    "MP": "Madhya Pradesh",
    "RJ": "Rajasthan",
    "PB": "Punjab",
    "HR": "Haryana",
    "UP": "Uttar Pradesh",
    "KA": "Karnataka",
    "TN": "Tamil Nadu",
    "AP": "Andhra Pradesh",
    "TS": "Telangana",
    "DL": "Delhi",
    "UK": "Uttarakhand",
    "BR": "Bihar",
    "JH": "Jharkhand",
    "OD": "Odisha",
    "WB": "West Bengal",
}

# 4 API Keys provided by user
DEFAULT_GEMINI_KEYS = [
    "AQ.Ab8RN6KWceXroDztcQYJDq2Pl9clzHf2wMWuN3N280aKNo6iPg",
    "AQ.Ab8RN6KEM_dUJkWLuRH7Oa2dUA7Vh-_RyPW-7vRUE4iLy5nY1A",
    "AQ.Ab8RN6LYJ5daRjNM73df-2mNvE5xFSVSuyb66oz9rZrbqfqxCQ",
    "AQ.Ab8RN6LYJ5daRjNM73df-2mNvE5xFSVSuyb66oz9rZrbqfqxCQ",
]

# Vision models supported with automatic fallback
SUPPORTED_VISION_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

GEMINI_SYSTEM_PROMPT = """You are an expert industrial computer vision model specialized in reading laser-cut metal stencils, stamp codes, and serial markings on agricultural implements and sheet metal parts.

The markings follow this standardized centralized pattern:
STATE / MFR_CODE / MODEL_CODE / YEAR / SERIAL [/ SUFFIX]
Example formats:
- Single-line: GJ/142/446/2026/11881/UP or GJ/142/445/2026/43889
- Two-line:
    Line 1: GJ/142/3674
    Line 2: /2026/10652
    (Combine both into GJ/142/3674/2026/10652)
- Without suffix: GJ/142/446/2026/11541 or GJ/142/3674/2026/10800

Catalog Reference:
- State Codes: GJ (Gujarat), MH (Maharashtra), MP (Madhya Pradesh), RJ (Rajasthan), UP (Uttar Pradesh), PB (Punjab), HR (Haryana), etc.
- Common Models: 444, 445, 446, 447, 3674, 3528
- Stencil bridge cut gaps: Characters are laser stencil cuts with bridge tabs. For example:
  - '446' can look like gaps in 4, 4, and 6 (never confuse with G, C, Q, 0).
  - '445' has gaps in 4 and 5 (never confuse with S).
  - '8' has a center bridge cut.
  - Years are typically 2026, 2025, 2024, 2023.
- Orientation & Mirroring:
  The metal part may be photographed upside down (180°), vertically (90°/270°), or mirrored from the rear of the through-cut metal plate. Carefully read the characters in their correct geometric reading order.
- Multi-Part / Reference Sheets:
  If the image is a paper or wall reference specification chart (e.g. titled "LIST OF CODES FOR LASER MARKING"), set is_document=true and list document_codes.
  If multiple stacked parts are present, put the clearest primary code in full_code, and list all distinct codes found in all_codes array.

Output MUST be strictly valid JSON matching this schema:
{
  "full_code": "GJ/142/446/2026/11881/UP",
  "state_code": "GJ",
  "mfr_code": "142",
  "model_code": "446",
  "year": "2026",
  "serial_no": "11881",
  "suffix": "UP",
  "is_valid": true,
  "is_document": false,
  "document_title": null,
  "document_codes": [],
  "all_codes": ["GJ/142/446/2026/11881/UP"],
  "orientation": "180deg upside down",
  "is_mirrored": false,
  "confidence": 0.99,
  "notes": "Laser-cut stencil marking oriented upside down on metal beam"
}
"""

class GeminiRoundRobinEngine:
    def __init__(self, api_keys: Optional[List[str]] = None, models: Optional[List[str]] = None):
        env_keys = os.environ.get("GEMINI_API_KEYS", "")
        if env_keys:
            self.api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        else:
            self.api_keys = list(api_keys or DEFAULT_GEMINI_KEYS)

        self.models = models or SUPPORTED_VISION_MODELS
        self._current_idx = 0
        self.stats = {
            "total_requests": 0,
            "success_count": 0,
            "failover_count": 0,
            "key_usage": [0] * len(self.api_keys)
        }

    def get_keys(self) -> List[str]:
        return self.api_keys

    def set_keys(self, keys: List[str]):
        if keys:
            self.api_keys = keys
            self.stats["key_usage"] = [0] * len(keys)
            self._current_idx = 0

    def _next_key_info(self) -> Tuple[str, int]:
        idx = self._current_idx % len(self.api_keys)
        key = self.api_keys[idx]
        self._current_idx = (self._current_idx + 1) % len(self.api_keys)
        return key, idx

    def generate_digital_stamp(self, code: str, parsed_info: Dict[str, Any]) -> np.ndarray:
        """
        Creates a high-contrast industrial label badge (White on Pure Black)
        displaying the detected implement code and metadata.
        """
        w, h = 760, 160
        stamp = np.zeros((h, w, 3), dtype=np.uint8)

        # Border
        cv2.rectangle(stamp, (4, 4), (w - 4, h - 4), (70, 70, 70), 2)
        cv2.rectangle(stamp, (8, 8), (w - 8, h - 8), (40, 40, 40), 1)

        # Title
        cv2.putText(stamp, "INDUSTRIAL LASER STENCIL IDENTIFIER", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        # Main Code text
        display_code = code if code else "NO CODE DETECTED"
        font_scale = 1.05 if len(display_code) <= 24 else 0.85
        cv2.putText(stamp, display_code, (20, 75),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # Model & State Subtitle
        m_code = parsed_info.get("model_code", "")
        model_name = IMPLEMENT_CATALOG.get(m_code, {}).get("name", "") if m_code else ""
        state_code = parsed_info.get("state_code", "")
        state_name = STATE_CODES.get(state_code, "") if state_code else ""

        desc_parts = []
        if state_name:
            desc_parts.append(f"State: {state_name} ({state_code})")
        if model_name:
            desc_parts.append(f"Model: {model_name}")
        desc_line = " | ".join(desc_parts)

        if desc_line:
            cv2.putText(stamp, desc_line, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (74, 222, 128), 1, cv2.LINE_AA)

        # Status badge
        is_valid = parsed_info.get("is_valid", False)
        is_doc = parsed_info.get("is_document", False)
        if is_doc:
            status_text = "[CATALOG CHART]"
            status_color = (250, 204, 21)
        elif is_valid:
            status_text = "[VALID STENCIL]"
            status_color = (74, 222, 128)
        else:
            status_text = "[RAW DETECTION]"
            status_color = (180, 180, 180)

        cv2.putText(stamp, status_text, (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

        # Engine Tag
        cv2.putText(stamp, "⚡ Powered by Gemini Vision AI (Round-Robin)", (20, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        return stamp

    def process_image(self, img_input: Any, timeout: int = 18) -> Dict[str, Any]:
        """
        Process image using Google Gemini Vision with round-robin key rotation
        and automatic failover across models and keys.
        """
        t0 = time.time()
        self.stats["total_requests"] += 1

        # Normalize image input to numpy BGR
        img_bgr = None

        if isinstance(img_input, str):
            img_bgr = cv2.imread(img_input)
            if img_bgr is None:
                # Handle Windows Unicode paths if needed
                try:
                    with open(img_input, "rb") as f:
                        file_bytes = np.frombuffer(f.read(), np.uint8)
                        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                except Exception:
                    pass
        elif isinstance(img_input, bytes):
            nparr = np.frombuffer(img_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif hasattr(img_input, "read"):
            nparr = np.frombuffer(img_input.read(), np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(img_input, np.ndarray):
            img_bgr = img_input

        if img_bgr is None:
            return {"success": False, "error": "Invalid or unreadable image input."}

        # Optimize size for fast upload & crisp vision OCR (downscale if > 1280)
        h, w = img_bgr.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            resized_for_api = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            resized_for_api = img_bgr

        _, buf = cv2.imencode(".jpg", resized_for_api, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = {
            "contents": [{
                "parts": [
                    {"text": GEMINI_SYSTEM_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json",
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        encoded_data = json.dumps(payload).encode("utf-8")

        total_keys = len(self.api_keys)
        last_error = None

        # Round-robin loop with multi-model fallback
        for attempt in range(total_keys):
            key, key_idx = self._next_key_info()

            for model_name in self.models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
                req = urllib.request.Request(
                    url,
                    data=encoded_data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )

                try:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        resp_json = json.loads(resp.read().decode("utf-8"))
                        raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                        parsed_res = json.loads(raw_text)

                        # Update metrics
                        self.stats["success_count"] += 1
                        self.stats["key_usage"][key_idx] += 1

                        # Extract fields
                        final_code = parsed_res.get("full_code", "").strip()
                        state_code = parsed_res.get("state_code", "")
                        model_code = parsed_res.get("model_code", "")

                        # Enhance parsed_info with local catalog knowledge
                        parsed_res["state_name"] = STATE_CODES.get(state_code, "")
                        parsed_res["model_info"] = IMPLEMENT_CATALOG.get(model_code, None)

                        # Generate digital stamp badge
                        stamp = self.generate_digital_stamp(final_code, parsed_res)

                        # Generate high-contrast binary cutout preview
                        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(gray)
                        _, bin_img = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        if np.mean(bin_img) > 127:
                            bin_img = cv2.bitwise_not(bin_img)

                        elapsed_s = round(time.time() - t0, 3)

                        return {
                            "success": True,
                            "engine": "gemini_vision",
                            "model_used": model_name,
                            "final_code": final_code,
                            "parsed_info": parsed_res,
                            "best_angle": parsed_res.get("orientation", "0deg"),
                            "is_mirrored": parsed_res.get("is_mirrored", False),
                            "key_used_slot": key_idx + 1,
                            "key_masked": f"...{key[-6:]}",
                            "attempts_made": attempt + 1,
                            "raw_detections": [{"text": final_code, "confidence": parsed_res.get("confidence", 0.99)}],
                            "digital_stamp": stamp,
                            "cropped_binary": bin_img,
                            "annotated_image": img_bgr,
                            "processing_time_s": elapsed_s,
                            "elapsed_ms": round(elapsed_s * 1000, 1)
                        }

                except Exception as e:
                    self.stats["failover_count"] += 1
                    last_error = e
                    # Try next model or next key in round-robin loop
                    continue

        return {
            "success": False,
            "error": f"All {total_keys} Gemini API keys failed. Last error: {str(last_error)}",
            "attempts_made": total_keys
        }
