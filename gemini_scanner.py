"""
Fast Gemini Vision Round-Robin OCR Engine for Laser Cut Metal Stencils.
Optimized for high-speed, minimal latency, and simple PARTno output.
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

SUPPORTED_VISION_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

def load_dotenv_if_exists():
    for env_path in [".env", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k and not os.environ.get(k):
                                os.environ[k] = v
            except Exception:
                pass

load_dotenv_if_exists()

FAST_SYSTEM_PROMPT = """You are an OCR model for laser-cut metal stencils on agricultural implement parts.
The stencil code follows: STATE/MFR/MODEL/YEAR/SERIAL[/SUFFIX]
Examples:
- GJ/142/446/2026/11881/UP
- GJ/142/3674/2026/10652
- GJ/142/445/2026/43889
The metal part might be upside down (180deg), vertical (90/270deg), or mirrored.
Return ONLY this exact JSON format:
{"PARTno": "GJ/142/446/2026/11881/UP"}
"""

class GeminiRoundRobinEngine:
    def __init__(self, api_keys: Optional[List[str]] = None, models: Optional[List[str]] = None):
        raw_env = (
            os.environ.get("GEMINI_API_KEYS", "")
            or os.environ.get("GEMINI_API_KEY", "")
            or ""
        )
        if raw_env:
            cleaned = []
            for k in raw_env.split(","):
                k = k.strip().strip('"').strip("'").strip()
                if k:
                    cleaned.append(k)
            self.api_keys = cleaned if cleaned else list(api_keys or [])
        else:
            self.api_keys = list(api_keys or [])

        self.models = models or SUPPORTED_VISION_MODELS
        self._current_idx = 0

    def get_keys(self) -> List[str]:
        return self.api_keys

    def _next_key_info(self) -> Tuple[str, int]:
        if not self.api_keys:
            raise ValueError("No Gemini API Keys configured. Please set GEMINI_API_KEYS environment variable.")
        idx = self._current_idx % len(self.api_keys)
        key = self.api_keys[idx]
        self._current_idx = (self._current_idx + 1) % len(self.api_keys)
        return key, idx

    def scan_part_number(self, img_input: Any, timeout: int = 15) -> Dict[str, Any]:
        """
        Fast execution path returning only the extracted PARTno and status.
        """
        if not self.api_keys:
            return {
                "status": "FAILED",
                "PARTno": None,
                "error": "No Gemini API keys configured. Set GEMINI_API_KEYS in Render/Vercel Environment Variables."
            }

        # Normalize image input to numpy BGR
        img_bgr = None
        if isinstance(img_input, str):
            img_bgr = cv2.imread(img_input)
            if img_bgr is None:
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
            return {
                "status": "FAILED",
                "PARTno": None,
                "error": "Invalid or unreadable image input."
            }

        # Resize to max 1024px for lightning-fast network upload & minimal latency
        h, w = img_bgr.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            resized = img_bgr

        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = {
            "contents": [{
                "parts": [
                    {"text": FAST_SYSTEM_PROMPT},
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
            try:
                key, key_idx = self._next_key_info()
            except Exception as e:
                last_error = e
                break

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

                        part_no = (
                            parsed_res.get("PARTno")
                            or parsed_res.get("part_no")
                            or parsed_res.get("full_code")
                            or ""
                        ).strip()

                        if part_no:
                            return {
                                "status": "SUCCESS",
                                "PARTno": part_no
                            }

                except Exception as e:
                    last_error = e
                    continue

        return {
            "status": "FAILED",
            "PARTno": None,
            "error": f"Scan failed across all keys: {str(last_error)}"
        }

    # Backward compatibility alias
    def process_image(self, img_input: Any, timeout: int = 15) -> Dict[str, Any]:
        res = self.scan_part_number(img_input, timeout=timeout)
        part_no = res.get("PARTno") or ""
        return {
            "success": res.get("status") == "SUCCESS",
            "status": res.get("status"),
            "final_code": part_no,
            "PARTno": part_no,
            "error": res.get("error")
        }
