"""
Industrial Laser Cut Metal Stencil OCR Engine.
Optimized for arbitrary orientations (0°, 90°, 180°, 270°), mirrored/backside through-cuts,
bridge-font restoration, multi-key round-robin, and instant minimal PARTno response.
"""

import os
import re
import json
import time
import base64
import urllib.request
import urllib.error
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List

SUPPORTED_VISION_MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
]

# Canonical catalog of authorized models and states
CATALOG_MODELS = {"444", "445", "446", "447", "3528", "3674"}

VALID_STATES = {
    "GJ", "MH", "MP", "RJ", "PB", "HR", "UP", "KA", "TN", "AP",
    "TS", "DL", "UK", "BR", "JH", "OD", "WB", "GA", "CH", "HP"
}

MODEL_ALIASES = {
    "44G": "446", "440": "446", "44Q": "446", "44C": "446", "44B": "446", "44F": "446",
    "4AG": "446", "4AC": "446", "4AF": "446", "4A6": "446", "4A0": "446", "4AB": "446",
    "N46": "446", "N4G": "446", "N4C": "446", "N4F": "446", "A46": "446", "A4G": "446",
    "AAG": "446", "AAC": "446", "AAF": "446", "NAG": "446", "NAC": "446", "NAF": "446",
    "M4G": "446", "M4F": "446", "IN4G": "446", "IA4G": "446", "INAG": "446", "INAC": "446",
    "NA.G": "446", "INA.G": "446", "AA4G": "446", "4G": "446", "44": "446", "4++": "446",
    "4+4+": "446", "44I": "446", "44E": "446", "A4G6": "446", "441": "446", "44L": "446",
    "4466": "446", "4406": "446", "42N46": "446", "MA4G": "446", "NA4G": "446", "MA46": "446",
    "NA46": "446", "4A4G": "446", "A44G": "446",
    "44S": "445", "4S": "445", "N45": "445", "A45": "445",
    "44A": "444", "4A": "444", "A44": "444", "N44": "444",
    "44T": "447", "4T": "447", "A47": "447", "N47": "447",
    "367A": "3674", "367N": "3674", "367H": "3674", "367": "3674",
    "352B": "3528", "352": "3528",
}

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

DUAL_VIEW_PROMPT = """You are an expert OCR model for laser-cut metal stencils on agricultural implement parts.
You are provided TWO views of the metal part:
- View 1: Original photo
- View 2: Horizontally flipped photo (un-mirrored, in case View 1 was taken from the reverse side of a through-cut plate)

The implement part number format: STATE/MFR/MODEL/YEAR/SERIAL[/SUFFIX]
- STATE: 2-letter state code (GJ, MH, MP, RJ, PB, HR, UP, etc.)
- MFR: 142
- MODEL: Must be one of: 444, 445, 446, 447, 3528, 3674
- YEAR: 4-digit year like 2024, 2025, 2026, 2027
- SERIAL: 5-digit number like 11881, 11541, 12313, 44579, 10800, 10652
- SUFFIX: Optional suffix like UP

CRITICAL INSTRUCTIONS:
1. One view is forward-readable (letters and numbers read normally, not backwards/mirrored). Compare View 1 and View 2 to find the forward-readable text.
2. Read from any rotation angle (0°, 90°, 180°, 270°).
3. If split across 2 lines (Line 1: GJ/142/3674, Line 2: 2026/10652), merge into a single code: GJ/142/3674/2026/10652.
4. Return ONLY JSON format:
{"PARTno": "GJ/142/446/2026/11881/UP"}
"""

def normalize_stencil_code(raw_code: str) -> Tuple[str, bool]:
    """
    Cleans, fixes stencil bridges, and validates standard implement syntax.
    Returns (cleaned_code, is_valid).
    """
    if not raw_code:
        return "", False

    code = raw_code.strip().replace(" ", "").replace("\\", "/").upper()
    code = re.sub(r"[^A-Z0-9/]", "", code)

    # Normalize state code bridges (e.g. CJ/ -> GJ/, CL/ -> GJ/)
    if code.startswith(("CJ/", "CL/", "GU/")):
        code = "GJ/" + code[3:]

    # Remove accidental leading 1 from /12026 -> /2026
    code = re.sub(r"/(1)(202[0-9])/", r"/\2/", code)

    # Standard pattern: STATE/MFR/MODEL/YEAR/SERIAL(/SUFFIX)?
    m = re.search(r"([A-Z]{2})/([0-9]{3})/([A-Z0-9]{2,5})/([0-9]{4})/([0-9]{5})(?:/([A-Z0-9]+))?", code)
    if m:
        state, mfr, model, year, serial, suffix = m.groups()
        # Normalize model bridge aliases
        if model in MODEL_ALIASES:
            model = MODEL_ALIASES[model]

        normalized = f"{state}/{mfr}/{model}/{year}/{serial}"
        if suffix:
            normalized += f"/{suffix}"

        is_valid = (
            state in VALID_STATES
            and model in CATALOG_MODELS
            and (2020 <= int(year) <= 2030)
            and len(serial) == 5
        )
        return normalized, is_valid

    # Check 2-part format if slash was replaced by 1 or missing
    m2 = re.search(r"([A-Z]{2})142([A-Z0-9]{3,4})(202[0-9])([0-9]{5})", code)
    if m2:
        state, model, year, serial = m2.groups()
        if model in MODEL_ALIASES:
            model = MODEL_ALIASES[model]
        normalized = f"{state}/142/{model}/{year}/{serial}"
        is_valid = (state in VALID_STATES and model in CATALOG_MODELS)
        return normalized, is_valid

    return code, False


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
        self._key_cooldowns: Dict[str, float] = {}

        # Optional local high-speed engine if installed
        self._local_engine = None
        try:
            from ocr_engine import StencilOCREngine
            self._local_engine = StencilOCREngine()
        except Exception:
            self._local_engine = None

    def get_keys(self) -> List[str]:
        return self.api_keys

    def _next_key_info(self) -> Tuple[str, int]:
        if not self.api_keys:
            raise ValueError("No Gemini API Keys configured. Please set GEMINI_API_KEYS environment variable.")
        now = time.time()
        n = len(self.api_keys)
        # 1. Look for a key that is not in 60s cooldown
        for _ in range(n):
            idx = self._current_idx % n
            key = self.api_keys[idx]
            self._current_idx = (self._current_idx + 1) % n
            if self._key_cooldowns.get(key, 0) <= now:
                return key, idx
        # 2. If all keys are currently cooling down, fallback to round-robin
        idx = self._current_idx % n
        key = self.api_keys[idx]
        self._current_idx = (self._current_idx + 1) % n
        return key, idx

    def _call_gemini_dual_view(self, img_bgr: np.ndarray, timeout: int = 12) -> Optional[str]:
        """
        Sends dual views (Original + Horizontally Flipped) to Gemini Vision.
        Handles all orientations (0°, 90°, 180°, 270°) and mirrored cuts in 1 call.
        """
        h, w = img_bgr.shape[:2]
        max_dim = 960
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            resized = img_bgr

        flipped = cv2.flip(resized, 1)

        _, buf1 = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        _, buf2 = cv2.imencode(".jpg", flipped, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_1 = base64.b64encode(buf1.tobytes()).decode("utf-8")
        b64_2 = base64.b64encode(buf2.tobytes()).decode("utf-8")

        payload = {
            "contents": [{
                "parts": [
                    {"text": DUAL_VIEW_PROMPT},
                    {"text": "View 1 (Original Photo):"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_1}},
                    {"text": "View 2 (Horizontally Flipped Photo):"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_2}}
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }
        encoded_data = json.dumps(payload).encode("utf-8")

        total_keys = len(self.api_keys)
        for attempt in range(total_keys):
            try:
                key, key_idx = self._next_key_info()
            except Exception:
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
                        raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if raw_text.startswith("```"):
                            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                            raw_text = re.sub(r"\s*```$", "", raw_text).strip()
                        parsed_res = json.loads(raw_text)

                        part_no = (
                            parsed_res.get("PARTno")
                            or parsed_res.get("part_no")
                            or parsed_res.get("full_code")
                            or ""
                        ).strip()

                        if part_no:
                            return part_no

                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        # Unauthorized/Invalid key: put in 1-hour cooldown & immediately switch to next key
                        self._key_cooldowns[key] = time.time() + 3600
                        break
                    # Model specific 404, 429, or 503: try next model on this key
                    continue
                except Exception:
                    continue

            # If all models failed on this key, put key in cooldown
            self._key_cooldowns[key] = time.time() + 60

        return None

    def scan_part_number(self, img_input: Any, timeout: int = 12) -> Dict[str, Any]:
        """
        Scans and extracts the exact implement PARTno.
        Response schema strictly matches:
        {"status": "SUCCESS", "PARTno": "..."}
        or {"status": "FAILED", "PARTno": null, "error": "..."}
        """
        # 1. Normalize image input to numpy BGR
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

        # 2. Fast Path: Local Stencil Engine (0.2s - if available and code is valid)
        if self._local_engine is not None:
            try:
                local_res = self._local_engine.process_image(img_bgr)
                code = local_res.get("final_code") or ""
                parsed_info = local_res.get("parsed_info") or {}
                if code and parsed_info.get("is_valid"):
                    norm_code, is_valid = normalize_stencil_code(code)
                    if is_valid:
                        return {
                            "status": "SUCCESS",
                            "PARTno": norm_code
                        }
            except Exception:
                pass

        # 3. Vision Path: Gemini Vision Dual-View (All Angles + Mirrored Cut)
        if self.api_keys:
            raw_gemini_code = self._call_gemini_dual_view(img_bgr, timeout=timeout)
            if raw_gemini_code:
                norm_code, is_valid = normalize_stencil_code(raw_gemini_code)
                return {
                    "status": "SUCCESS",
                    "PARTno": norm_code if norm_code else raw_gemini_code
                }

        # 4. Fallback check on local engine even if is_valid was False
        if self._local_engine is not None:
            try:
                local_res = self._local_engine.process_image(img_bgr)
                code = local_res.get("final_code") or ""
                if code:
                    norm_code, _ = normalize_stencil_code(code)
                    return {
                        "status": "SUCCESS",
                        "PARTno": norm_code if norm_code else code
                    }
            except Exception:
                pass

        return {
            "status": "FAILED",
            "PARTno": None,
            "error": "No stencil code could be recognized. Please ensure metal part is visible with clear lighting."
        }

    # Backward compatibility alias
    def process_image(self, img_input: Any, timeout: int = 12) -> Dict[str, Any]:
        return self.scan_part_number(img_input, timeout=timeout)
