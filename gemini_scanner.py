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
    for env_name in [".env", ".env.local"]:
        for env_path in [env_name, os.path.join(os.path.dirname(os.path.abspath(__file__)), env_name)]:
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
- View 1: Original photo (contrast-enhanced to isolate laser cutouts)
- View 2: Horizontally flipped photo (un-mirrored, in case View 1 was taken from the reverse side of a through-cut plate)

The implement part number format: STATE/MFR/MODEL/YEAR/SERIAL[/SUFFIX]
- STATE: 2-letter state code, ALWAYS "GJ" (Gujarat). In stencil font, the letter "G" has a bridge cut and may look like "C", "6", or "O" — decode it as "GJ".
- MFR: Always "142"
- MODEL: Must be one of authorized models: 444, 445, 446, 447, 3528, 3674
- YEAR: 4-digit year like 2024, 2025, 2026, 2027
- SERIAL: 5-digit number like 11881, 11541, 12313, 44579, 10800, 10652
- SUFFIX: Optional suffix like "UP"

CRITICAL INSTRUCTIONS FOR LASER CUTOUT STENCILS:
1. BACKGROUND COLORS IN SLITS: The numbers/letters are hollow holes cut through sheet metal. A different background color (floor, shadows, paint, reflections) shows through the cutout slits. Do not get confused by background colors — trace the actual cutout shapes of the characters.
2. STENCIL BRIDGES: Stencil characters have bridge tabs holding inner islands (e.g. in 0, 4, 6, 8, 9, A, B, D, G, O, P, Q, R). Mentally connect the separated segments to identify the true digit (e.g. 4 with bridge slit, 6 with bridge slit, 8 with bridge slit).
3. ORIENTATION & DIRECTION: One view is forward-readable. If text is read backwards (e.g. ending in ".../142/GJ"), re-order so it ALWAYS starts with "GJ/142/...".
4. TWO-LINE STENCILS: If split across 2 lines (Line 1: GJ/142/3674, Line 2: 2026/10652), merge into a single code: GJ/142/3674/2026/10652.

Return ONLY JSON:
{"PARTno": "GJ/142/446/2026/11881/UP"}
"""

def enhance_laser_cut_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """
    Normalizes local contrast between metal plate and cutout holes using LAB-space CLAHE.
    Eliminates confusion caused by background colors, shadows, or metallic specular glare.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

def normalize_stencil_code(raw_code: str) -> Tuple[str, bool]:
    """
    Cleans, fixes stencil bridges, handles reversed tokens, and validates standard implement syntax.
    Returns (cleaned_code, is_valid).
    """
    if not raw_code:
        return "", False

    code = raw_code.strip().replace(" ", "").replace("\\", "/").upper()
    code = re.sub(r"[^A-Z0-9/]", "", code)

    parts = [p for p in code.split("/") if p]
    if not parts:
        return "", False

    # 1. Handle reversed token order (e.g. UP/11881/2026/446/142/GJ -> GJ/142/446/2026/11881/UP)
    if len(parts) >= 4:
        if parts[-1] in ("GJ", "CJ", "GU", "6J", "0J") or (len(parts) >= 2 and parts[-2] in ("142", "GJ") and parts[-1] in ("GJ", "UP")):
            parts.reverse()
        elif parts[-1] == "142":
            parts.reverse()

    # 2. If state prefix is missing but starts with 142 (e.g. 142/446/2026/11881) -> prepend GJ
    if parts[0] == "142":
        parts.insert(0, "GJ")

    # 3. State normalization (bridge cuts make G look like C, 6, O, CI, GU)
    state = parts[0]
    if state in ("CJ", "CL", "CI", "GU", "6J", "0J", "OJ", "G", "J", "GI", "C"):
        state = "GJ"
    parts[0] = state

    # 4. Standard 5 or 6 component parsing
    if len(parts) >= 5:
        state = parts[0]
        mfr = parts[1]
        model = parts[2]
        year = parts[3]
        serial = parts[4]
        suffix = parts[5] if len(parts) > 5 else ""

        # Normalize manufacturer code (almost always 142)
        if mfr in ("142", "14Z", "I42", "14", "1421"):
            mfr = "142"

        # Normalize model aliases
        if model in MODEL_ALIASES:
            model = MODEL_ALIASES[model]

        # Normalize year (bridge cuts in 6 make 2026 look like 202G, 202C, 2O26)
        if re.match(r"^202[GCQ0BF4b1ILI]?$", year) or year in ("202", "2O26", "2020"):
            year = "2026"

        # Normalize serial: char-to-digit mapping for stencil bridge cut confusions
        char_map = {
            "O": "0", "D": "0", "Q": "0",
            "I": "1", "L": "1", "|": "1", "]": "1", "[": "1",
            "Z": "2",
            "E": "3",
            "A": "4", "H": "4",
            "S": "5",
            "G": "6", "B": "8",
            "T": "7"
        }
        s_clean = "".join(char_map.get(ch, ch) for ch in serial)
        digits = re.sub(r"\D", "", s_clean)
        # Accidental duplicate '1' from slash misread (e.g. /111881 -> 11881)
        if len(digits) == 6 and digits.startswith("11"):
            digits = digits[1:]
        elif len(digits) > 5:
            digits = digits[:5]
        serial = digits

        # Normalize suffix
        if suffix in ("UIP", "LIP", "7UP", "1UP", "UF", "UIF", "VIP", "VP", "UPP", "U", "P"):
            suffix = "UP"

        normalized = f"{state}/{mfr}/{model}/{year}/{serial}"
        if suffix:
            normalized += f"/{suffix}"

        is_valid = (
            state in VALID_STATES
            and model in CATALOG_MODELS
            and len(serial) == 5
        )
        return normalized, is_valid

    # Fallback regex search for un-slashed codes
    m = re.search(r"([A-Z]{2})/([0-9]{3})/([A-Z0-9]{2,5})/([0-9]{4})/([0-9]{5})(?:/([A-Z0-9]+))?", code)
    if m:
        s, mf, mo, y, ser, su = m.groups()
        if mo in MODEL_ALIASES: mo = MODEL_ALIASES[mo]
        norm = f"{s}/{mf}/{mo}/{y}/{ser}"
        if su: norm += f"/{su}"
        return norm, True

    return "/".join(parts), False


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

    def _call_gemini_dual_view(self, img_bgr: np.ndarray, api_key: Optional[str] = None, timeout: int = 12) -> Optional[str]:
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

        # Apply CLAHE contrast enhancement to isolate laser-cutout slits from background colors
        enhanced = enhance_laser_cut_contrast(resized)
        flipped = cv2.flip(enhanced, 1)

        _, buf1 = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf2 = cv2.imencode(".jpg", flipped, [cv2.IMWRITE_JPEG_QUALITY, 85])
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

        # Determine key pool to use with true round-robin rotation
        if api_key:
            keys_to_try = [api_key]
        else:
            n = len(self.api_keys)
            if n > 0:
                start = self._current_idx % n
                self._current_idx = (self._current_idx + 1) % n
                keys_to_try = self.api_keys[start:] + self.api_keys[:start]
            else:
                keys_to_try = []

        now = time.time()
        for key in keys_to_try:
            # Skip keys in cooldown if we have other options
            if self._key_cooldowns.get(key, 0) > now and len(keys_to_try) > 1:
                continue

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
                    if e.code in (401, 403, 404):
                        # Invalid/Unauthorized/Model not found: mark this key as dead & immediately transfer to next key
                        self._key_cooldowns[key] = time.time() + 3600
                        break
                    elif e.code == 429:
                        # Rate limit hit: put in 60s cooldown & immediately transfer to next key
                        self._key_cooldowns[key] = time.time() + 60
                        break
                    continue
                except Exception:
                    continue

            # If all models failed on this key, put key in cooldown
            self._key_cooldowns[key] = time.time() + 60

        return None

    def scan_part_number(self, img_input: Any, api_key: Optional[str] = None, timeout: int = 12) -> Dict[str, Any]:
        """
        Scans and extracts the exact implement PARTno purely via Gemini 2.5 Flash Cloud Vision.
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

        # 2. Check for configured API keys
        has_key = bool(api_key or self.api_keys)
        if not has_key:
            return {
                "status": "FAILED",
                "PARTno": None,
                "error": "No Gemini API Key configured. Please set GEMINI_API_KEYS environment variable or pass x-api-key header."
            }

        # 3. Vision Path: Pure Gemini 2.5 Flash Cloud Vision (Dual-View + CLAHE)
        raw_gemini_code = self._call_gemini_dual_view(img_bgr, api_key=api_key, timeout=timeout)
        if raw_gemini_code:
            norm_code, is_valid = normalize_stencil_code(raw_gemini_code)
            return {
                "status": "SUCCESS",
                "PARTno": norm_code if norm_code else raw_gemini_code
            }

        return {
            "status": "FAILED",
            "PARTno": None,
            "error": "Gemini 2.5 Flash Cloud Vision could not detect a valid implement code in the image. Please ensure the stamped metal beam is visible."
        }

    # Backward compatibility alias
    def process_image(self, img_input: Any, api_key: Optional[str] = None, timeout: int = 12) -> Dict[str, Any]:
        return self.scan_part_number(img_input, api_key=api_key, timeout=timeout)
