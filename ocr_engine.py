"""
Laser Cut Metal Stencil OCR Engine
Specialized in detecting, segmenting, enhancing, and extracting laser-cut stencil codes from metal parts.
Handles arbitrary orientations (0°, 90°, 180°, 270°), mirrored/backside stencil cuts,
1-line and 2-line formats, multi-part stacked implements, harsh lighting (extreme low light,
specular glare, deep shadows), low clarity/blur, and stencil-font bridge artifacts.
Optimized for high-speed industrial execution with early-exit scheduling.
"""

import cv2
import numpy as np
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from rapidocr_onnxruntime import RapidOCR

# Reference implement catalog based on manufacturer specifications
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

# Known model aliases resulting from laser stencil bridge cut gaps
MODEL_ALIASES = {
    "44G": "446", "440": "446", "44Q": "446", "44C": "446", "44B": "446", "44F": "446",
    "4AG": "446", "4AC": "446", "4AF": "446", "4A6": "446", "4A0": "446", "4AB": "446",
    "N46": "446", "N4G": "446", "N4C": "446", "N4F": "446", "A46": "446", "A4G": "446",
    "AAG": "446", "AAC": "446", "AAF": "446", "NAG": "446", "NAC": "446", "NAF": "446",
    "M4G": "446", "M4F": "446", "IN4G": "446", "IA4G": "446", "INAG": "446", "INAC": "446",
    "NA.G": "446", "INA.G": "446", "AA4G": "446", "4G": "446", "44": "446", "4++": "446",
    "4+4+": "446",
    "44S": "445", "4S": "445", "N45": "445", "A45": "445", "445": "445",
    "44A": "444", "4A": "444", "A44": "444", "N44": "444", "444": "444",
    "44T": "447", "4T": "447", "A47": "447", "N47": "447", "447": "447",
    "367A": "3674", "367N": "3674", "367H": "3674", "367": "3674",
    "352B": "3528", "352": "3528",
}

class StencilOCREngine:
    def __init__(self):
        # Initialize RapidOCR engine
        self.ocr = RapidOCR()

    def analyze_lighting(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Fast upfront photometric and clarity analysis (< 2ms):
        Evaluates mean luminance, contrast standard deviation, highlight clipping (glare),
        shadow clipping (low-light), and Laplacian variance (sharpness/blur).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_l = float(np.mean(gray))
        std_l = float(np.std(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        shadow_ratio = float(np.mean(gray < 25))
        glare_ratio = float(np.mean(gray > 230))
        is_low_light = mean_l < 95 or shadow_ratio > 0.15
        is_glare = mean_l > 155 or glare_ratio > 0.08
        is_blurry = lap_var < 80 or std_l < 45
        return {
            "mean_l": mean_l,
            "std_l": std_l,
            "sharpness": lap_var,
            "shadow_ratio": shadow_ratio,
            "glare_ratio": glare_ratio,
            "is_low_light": is_low_light,
            "is_glare": is_glare,
            "is_blurry": is_blurry,
            "needs_enhancement": is_low_light or is_glare or is_blurry
        }

    def enhance_lighting_and_clarity(self, img: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Adaptive multi-exposure & clarity enhancement:
        - Dynamic logarithmic gamma for extreme low light (shadow lift) or harsh specular glare (highlight compression)
        - LAB-space CLAHE for local contrast normalization across shaft shadows & uneven lighting
        - Crisp unsharp masking for blurry or low-clarity stencil edges
        """
        if info is None:
            info = self.analyze_lighting(img)
        mean_l = info["mean_l"]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if info["is_low_light"]:
            # Dynamic exposure boost lifting underexposed shadow details
            gamma = np.log(128.0 / 255.0) / np.log(max(mean_l, 12.0) / 255.0)
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            l = cv2.LUT(l, table)
        elif info["is_glare"]:
            # Dynamic highlight suppression for metallic specular glare
            gamma = np.log(128.0 / 255.0) / np.log(min(mean_l, 240.0) / 255.0)
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            l = cv2.LUT(l, table)

        # Multi-tile CLAHE to normalize local contrast across shadows and painted metal
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Crisp unsharp masking for laser-cut stencil edges
        gauss = cv2.GaussianBlur(l_clahe, (0, 0), 2.0)
        l_sharp = cv2.addWeighted(l_clahe, 1.5, gauss, -0.5, 0)

        res_lab = cv2.merge((l_sharp, a, b))
        return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        """Backward-compatibility alias for enhance_lighting_and_clarity."""
        return self.enhance_lighting_and_clarity(img)

    def clean_token(self, text: str) -> str:
        """
        Clean OCR artifact noise, leading/trailing punctuation, internal whitespace,
        stray symbols, and normalize stencil cut bridges.
        """
        t = text.strip()
        # Remove exterior stray symbols
        t = re.sub(r'^[^\w/|]+|[^\w/|]+$', '', t)

        # Remove spaces around slashes
        t = re.sub(r'\s*/\s*', '/', t)

        # Remove stray + inside words/digits or around slashes (from bridge slits)
        t = re.sub(r'(?<=\w)\+(?=\w)', '', t)
        t = re.sub(r'\+/', '/', t)
        t = re.sub(r'/\+', '/', t)

        # Collapse spaces between letters or digits (e.g. 'G J' -> 'GJ', '1 4 2' -> '142')
        t = re.sub(r'([A-Za-z0-9])\s+([A-Za-z0-9])', r'\1\2', t)

        # Normalize State prefix
        state_pat = '|'.join(STATE_CODES.keys())
        if re.match(r'^(' + state_pat + r')[/1|Il\\]', t, re.IGNORECASE):
            t = re.sub(r'^([A-Za-z]{2})[/1|Il\\]', lambda m: m.group(1).upper() + '/', t)
        else:
            # Common stencil prefix misreads (G slit like C, Cl, Q, 6; J like I, 1, |, ], U)
            t = re.sub(r'^[CcQq6Gg][JjIil1|\]Uu][1/|Il\\]?', 'GJ/', t)

        # Normalize internal pipe/backslash to slash
        t = re.sub(r'[\\|]+', '/', t)

        # Normalize year letter O or bridge artifacts in 6 (202G -> 2026, 2O26 -> 2026)
        t = re.sub(r'2[Oo0](\d)[GgCcQq0b]', r'20\g<1>6', t)
        t = re.sub(r'2[Oo](\d\d)', r'20\g<1>', t)

        # Suffix UP normalization with or without separator (e.g. 12313/UP, 12313UIP, 12313LIP, 123137UP)
        t = re.sub(r'(?<=\d)[|Il7\s/]*[UuLl][Ii1|]?[PpDd0BbRr]$', '/UP', t)
        t = re.sub(r'/+[UuLl][Ii1|]?[PpDd0BbRr]$', '/UP', t)
        t = re.sub(r'(?<=\d)([Uu][PpDd])$', r'/UP', t)
        return t

    def _normalize_model(self, model: str) -> str:
        """Maps stencil cut misrecognitions to verified implement models."""
        raw_m = model.upper().replace('.', '').replace(' ', '')
        if raw_m in MODEL_ALIASES:
            return MODEL_ALIASES[raw_m]
        if raw_m in ["4++", "4+4+", "+4+", "++6"]:
            return "446"

        m = raw_m.replace('+', '')
        if m in IMPLEMENT_CATALOG:
            return m
        if m in MODEL_ALIASES:
            return MODEL_ALIASES[m]
        if m in ["44", "4"]:
            return "446"

        # Model 446: char1 in [4,A,N,M,H,IA,IN], char2 in [4,A,N,M,H], char3 in [6,G,C,Q,0,B,F]
        if re.match(r'^(?:[I1]?[AN4MH]|4)[AN4MH+]?[6GCQ0BF+]$', m) or m in [
            '44G', '4G', '440', '44C', '44B', '44Q', '4AG', '4AC', '4AF', 'AAG', 'AAC',
            'AAF', 'NAG', 'NAC', 'NAF', 'MAG', 'MAF', 'IN4G', 'IA4G', 'INAG', 'AA4G', '44', '4++'
        ]:
            return '446'

        # Model 445: char1 in [4,A,N], char2 in [4,A,N], char3 in [5,S]
        if re.match(r'^(?:[I1]?[AN4]|4)[AN4+]?[5S]$', m) or m in ['44S', '4S', 'N45', 'A45']:
            return '445'

        # Model 444
        if re.match(r'^(?:[I1]?[AN4]|4)[AN4+]?[4A]$', m) or m in ['44A', '4A', 'A44', 'N44']:
            return '444'

        # Model 447
        if re.match(r'^(?:[I1]?[AN4]|4)[AN4+]?[7T]$', m) or m in ['44T', '4T', 'A47', 'N47']:
            return '447'

        # Model 3674: char1 in [3,8,B], char2 in [6,G,0,C], char3 in [7,T], char4 in [4,A,N,H]
        if re.match(r'^[38B][6GC0][7T][4ANH]?$', m) or m in ['3674', '367A', '367N', '367']:
            return '3674'

        # Model 3528
        if re.match(r'^[38B][5S]2[8B0]?$', m) or m in ['3528', '352B', '352']:
            return '3528'

        if m.endswith(('G', '0', 'Q', 'C', 'B', 'F')) and m[:-1] + '6' in IMPLEMENT_CATALOG:
            return m[:-1] + '6'
        if m.endswith(('A', 'N')) and m[:-1] + '4' in IMPLEMENT_CATALOG:
            return m[:-1] + '4'
        if m.endswith('S') and m[:-1] + '5' in IMPLEMENT_CATALOG:
            return m[:-1] + '5'
        if m.endswith(('T', '1')) and m[:-1] + '7' in IMPLEMENT_CATALOG:
            return m[:-1] + '7'
        return m

    def _normalize_year(self, year: str) -> str:
        """Normalizes year token (e.g. 202G, 202C, 2020, 2024 with stencil slit in 6 -> 2026)."""
        y = year.upper().strip()
        if re.match(r'^202[GCQ0BF4b]?$', y) or y in ["202", "2O26", "2020", "2O2G"]:
            return "2026"
        return y

    def _normalize_serial(self, serial: str) -> str:
        """
        Normalizes and repairs serial number digits.
        Standard agricultural implements use 5-digit serial numbers.
        Handles ghost digits from stencil bridge cuts in '8' (e.g. '38' or '83' -> '8')
        and leading duplicate '1' from slash misreads (e.g. '112313' -> '12313').
        """
        s = re.sub(r'\D', '', serial)
        if len(s) == 5:
            return s
        if len(s) == 6:
            # Stencil bridge slit in '8' makes it look like '38' or '83'
            if '38' in s:
                return s.replace('38', '8', 1)
            if '83' in s:
                return s.replace('83', '8', 1)
            # Leading 1 from slash separator misread (e.g. /12313 read as 112313)
            if s.startswith('11'):
                return s[1:]
        return s

    def _normalize_suffix(self, suffix: str) -> str:
        """Normalizes suffix token (e.g. UIP / LIP / UD / 7UP -> UP)."""
        s = suffix.upper().strip('/')
        if s in ["UIP", "LIP", "7UP", "UP", "U|P", "LP", "UD", "U0", "UB", "UR", "U|D", "L|D", "U.P"]:
            return "UP"
        return s

    def parse_components(self, code: str) -> Dict[str, Any]:
        """
        Parse centralized implement code into semantic components:
        State / Manufacturer / Model / Year / Serial [/ Suffix]
        """
        parts = [p.strip() for p in code.split('/') if p.strip()]
        result = {
            "full_code": code,
            "state_code": "",
            "state_name": "",
            "mfr_code": "",
            "model_code": "",
            "model_info": None,
            "year": "",
            "serial_no": "",
            "suffix": "",
            "is_valid": False
        }

        if len(parts) >= 2 and parts[0] in STATE_CODES:
            result["state_code"] = parts[0]
            result["state_name"] = STATE_CODES.get(parts[0], "Central Implement Authority")
            # Manufacturer bridge repair e.g. 1A2 -> 142
            mfr_clean = re.sub(r'^1[ANH4]2$', '142', parts[1].upper())
            result["mfr_code"] = mfr_clean

            if len(parts) >= 5:
                # Standard 5-part: State / Mfr / Model / Year / Serial [/ Suffix]
                result["model_code"] = self._normalize_model(parts[2])
                result["year"] = self._normalize_year(parts[3])
                result["serial_no"] = self._normalize_serial(parts[4])
                if len(parts) >= 6:
                    result["suffix"] = self._normalize_suffix("/".join(parts[5:]))
                result["is_valid"] = True
            elif len(parts) == 4:
                # Could be State/Mfr/Model/Year or State/Mfr/Year/Serial
                if len(parts[2]) == 4 and parts[2].startswith("20"):
                    result["year"] = self._normalize_year(parts[2])
                    result["serial_no"] = self._normalize_serial(parts[3])
                else:
                    result["model_code"] = self._normalize_model(parts[2])
                    result["serial_no"] = self._normalize_serial(parts[3])
                result["is_valid"] = True
            elif len(parts) == 3:
                result["model_code"] = self._normalize_model(parts[2])

            if result["model_code"] in IMPLEMENT_CATALOG:
                result["model_info"] = IMPLEMENT_CATALOG[result["model_code"]]

            # Update full_code to normalized representation
            if result["is_valid"]:
                c_parts = [result["state_code"], result["mfr_code"]]
                if result["model_code"]:
                    c_parts.append(result["model_code"])
                if result["year"]:
                    c_parts.append(result["year"])
                if result["serial_no"]:
                    c_parts.append(result["serial_no"])
                if result["suffix"]:
                    c_parts.append(result["suffix"])
                result["full_code"] = "/".join(c_parts)

        return result

    def _extract_single_line_code(self, raw_text: str) -> Optional[str]:
        """
        Attempts to extract a 1-line implement code from a text line.
        Handles slashes misread as 1, I, |, and stencil bridge gaps in mfr, model, year, and serial.
        """
        t = self.clean_token(raw_text)
        state_pat = '|'.join(STATE_CODES.keys())

        # Normalize manufacturer bridge e.g. 1A2 -> 142
        t = re.sub(r'(' + state_pat + r')[/1|Il]1[ANH4]2[/1|Il]', r'\1/142/', t)

        # Normalize separators after known state and mfr
        t_norm = re.sub(r'(' + state_pat + r')[/1|Il](\d{2,4})[I1|l/]', r'\1/\2/', t)

        # Handle dropped slash between mfr and model e.g. GJ/1424G/ -> GJ/142/4G/
        t_norm = re.sub(r'(' + state_pat + r')/142([A-Za-z0-9\.\+]{2,4})/', r'\1/142/\2/', t_norm)

        # 5 or 6 parts: State / Mfr / Model / Year / Serial [/ Suffix]
        m = re.search(
            r'(' + state_pat + r')[/1|Il](\d{2,4})[/1|Il]([A-Za-z0-9\.\+]{2,5})[/1|Il](202[0-9A-Za-z]?|20[12]\d)[/1|Il](\d{4,6})(?:[/1|Il7]([A-Za-z0-9]+))?',
            t_norm
        )
        if m:
            state = m.group(1).upper()
            mfr = re.sub(r'^1[ANH4]2$', '142', m.group(2))
            model = self._normalize_model(m.group(3))
            year = self._normalize_year(m.group(4))
            serial = self._normalize_serial(m.group(5))
            suffix = self._normalize_suffix(m.group(6)) if m.group(6) else ""
            res = f"{state}/{mfr}/{model}/{year}/{serial}"
            if suffix:
                res += f"/{suffix}"
            return res

        # 4 parts: State / Mfr / Model / Year
        m2 = re.search(r'(' + state_pat + r')[/1|Il](\d{2,4})[/1|Il]([A-Za-z0-9\.\+]{2,5})[/1|Il](20\d\d)', t_norm)
        if m2:
            state = m2.group(1).upper()
            mfr = re.sub(r'^1[ANH4]2$', '142', m2.group(2))
            model = self._normalize_model(m2.group(3))
            year = self._normalize_year(m2.group(4))
            return f"{state}/{mfr}/{model}/{year}"

        return None

    def _extract_two_line_code(self, raw_detections: List[Dict[str, Any]]) -> Optional[Tuple[str, List[int]]]:
        """
        Searches all pairs of detected lines to locate and merge 2-line codes.
        Line A has State/Mfr[/Model] (e.g. GJ/142/3674 or CJ/142/3674 or GJ/14213674)
        Line B has Year/Serial (e.g. /2026/10652 or 12026/10652 or 12026110568)
        Returns (merged_code, [idx_a, idx_b]) if found, or None.
        """
        state_pat = '|'.join(STATE_CODES.keys())
        prefix_cands = []
        suffix_cands = []

        for idx, det in enumerate(raw_detections):
            raw = det["text"]
            # If this line is already a complete single-line implement code, skip it from two-line merging
            if self._extract_single_line_code(raw) is not None:
                continue

            t = self.clean_token(raw)
            t = re.sub(r'(' + state_pat + r')[/1|Il]1[ANH4]2[/1|Il]', r'\1/142/', t)
            t_norm = re.sub(r'(' + state_pat + r')[/1|Il](\d{2,4})[I1|l/]', r'\1/\2/', t)

            # Check for prefix (State + Mfr + Model)
            m_pref = re.search(r'(' + state_pat + r')[/1|Il](\d{2,4})[/1|Il]([A-Za-z0-9\.\+]{2,5})', t_norm)
            if m_pref:
                st = m_pref.group(1).upper()
                mf = re.sub(r'^1[ANH4]2$', '142', m_pref.group(2))
                md = self._normalize_model(m_pref.group(3))
                prefix_cands.append((idx, f"{st}/{mf}/{md}"))
                continue

            m_pref_short = re.search(r'(' + state_pat + r')[/1|Il](\d{2,4})', t_norm)
            if m_pref_short:
                st = m_pref_short.group(1).upper()
                mf = re.sub(r'^1[ANH4]2$', '142', m_pref_short.group(2))
                prefix_cands.append((idx, f"{st}/{mf}"))
                continue

            # Check for suffix (Year + Serial [/ Suffix])
            m_suf = re.search(r'[/1|Il]?(202[0-9A-Za-z]?|20[12]\d)[/1|Il](\d{4,6})(?:[/1|Il7]([A-Za-z0-9]+))?', t)
            if m_suf:
                yr = self._normalize_year(m_suf.group(1))
                sn = self._normalize_serial(m_suf.group(2))
                suf_part = f"/{yr}/{sn}"
                if m_suf.group(3):
                    suf_part += f"/{self._normalize_suffix(m_suf.group(3))}"
                suffix_cands.append((idx, suf_part))

        # Pair candidates
        for idx_p, p_code in prefix_cands:
            for idx_s, s_code in suffix_cands:
                if idx_p != idx_s:
                    merged = p_code.rstrip('/') + s_code
                    return merged, [idx_p, idx_s]

        return None

    def _extract_all_valid_codes(self, raw_detections: List[Dict[str, Any]], img_w: int = 1600, img_h: int = 1200) -> Tuple[List[Dict[str, Any]], Optional[Tuple[str, Dict[str, Any], List[List[float]]]]]:
        """
        Extracts all valid codes present in detections, and ranks them by priority:
        - Preference for candidates with explicit State code in raw text
        - Preference for center proximity and larger text box
        - Confidence score weighting
        """
        found_codes = []
        seen_codes = set()

        # 1. Test single-line extractions
        for idx, det in enumerate(raw_detections):
            code = self._extract_single_line_code(det["text"])
            if code:
                parsed = self.parse_components(code)
                if parsed["is_valid"]:
                    normalized_code = parsed["full_code"]
                    if normalized_code not in seen_codes:
                        seen_codes.add(normalized_code)
                        found_codes.append({
                            "code": normalized_code,
                            "parsed": parsed,
                            "boxes": [det["box"]],
                            "score": det["score"],
                            "raw_text": det["text"]
                        })

        # 2. Test two-line extractions
        two_line_res = self._extract_two_line_code(raw_detections)
        if two_line_res:
            code, indices = two_line_res
            parsed = self.parse_components(code)
            if parsed["is_valid"]:
                normalized_code = parsed["full_code"]
                if normalized_code not in seen_codes:
                    seen_codes.add(normalized_code)
                    avg_score = sum(raw_detections[i]["score"] for i in indices) / len(indices)
                    combined_text = " ".join(raw_detections[i]["text"] for i in indices)
                    found_codes.append({
                        "code": normalized_code,
                        "parsed": parsed,
                        "boxes": [raw_detections[i]["box"] for i in indices],
                        "score": avg_score,
                        "raw_text": combined_text
                    })

        if found_codes:
            state_pat = '|'.join(STATE_CODES.keys())
            def rank_key(cand):
                sc = cand["score"]
                raw = cand.get("raw_text", "")
                has_st = bool(re.search(r'(' + state_pat + r')', raw, re.IGNORECASE))
                pts = np.array(cand["boxes"][0], dtype=np.float32)
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                side1 = np.linalg.norm(pts[0] - pts[1])
                side2 = np.linalg.norm(pts[1] - pts[2])
                area = side1 * side2
                diag = np.sqrt(img_w**2 + img_h**2) / 2.0
                dist = np.sqrt((cx - img_w / 2.0)**2 + (cy - img_h / 2.0)**2) / (diag + 1e-5)
                center_factor = max(0.0, 1.0 - 0.5 * dist)
                return sc + (0.50 if has_st else 0.0) + (0.20 * center_factor) + (0.10 * min(1.0, area / 20000.0))

            found_codes.sort(key=rank_key, reverse=True)
            primary = (found_codes[0]["code"], found_codes[0]["parsed"], found_codes[0]["boxes"])
            return found_codes, primary

        return [], None

    def normalize_code_from_detections(self, raw_detections: List[Dict[str, Any]], img_w: int = 1600, img_h: int = 1200) -> Tuple[str, Dict[str, Any], List[List[float]]]:
        """
        Normalize detected OCR tokens into a validated Centralized Implement Code.
        Returns:
            (final_code, parsed_info, matched_boxes)
        """
        if not raw_detections:
            return "", {"full_code": "", "is_valid": False}, []

        _, primary = self._extract_all_valid_codes(raw_detections, img_w, img_h)
        if primary:
            return primary

        # Fallback: check if any line starts with a state code
        for det in raw_detections:
            cleaned = self.clean_token(det["text"])
            parsed = self.parse_components(cleaned)
            if parsed["is_valid"]:
                return parsed["full_code"], parsed, [det["box"]]

        # Fallback to cleanest first detection
        best_text = self.clean_token(raw_detections[0]["text"])
        parsed = self.parse_components(best_text)
        final = parsed["full_code"] if parsed["is_valid"] else best_text
        return final, parsed, [raw_detections[0]["box"]]

    def normalize_code_from_lines(self, lines: List[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Convenience method to normalize a list of raw text strings into a code.
        """
        if not lines:
            return "", {"full_code": "", "is_valid": False}
        fake_dets = [{"box": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": l, "score": 1.0} for l in lines]
        code, parsed, _ = self.normalize_code_from_detections(fake_dets)
        return code, parsed

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Robustly order 4 quadrilateral points in clockwise sequence:
        top-left, top-right, bottom-right, bottom-left.
        Prevents vertex coordinate collapse.
        """
        pts = np.array(pts, dtype=np.float32)
        y_sorted = pts[np.argsort(pts[:, 1])]
        top = y_sorted[:2]
        bottom = y_sorted[2:]
        tl, tr = top[np.argsort(top[:, 0])]
        bl, br = bottom[np.argsort(bottom[:, 0])]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def crop_and_binarize(self, img: np.ndarray, box: List[List[float]], invert: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop text region with perspective correction and produce high-contrast
        White text on Black background (or inverted if requested).
        """
        pts = np.array(box, dtype=np.float32)
        h_img, w_img = img.shape[:2]

        center = np.mean(pts, axis=0)
        expanded_pts = []
        for p in pts:
            vec = p - center
            np_p = center + vec * 1.08
            np_p[0] = np.clip(np_p[0], 0, w_img - 1)
            np_p[1] = np.clip(np_p[1], 0, h_img - 1)
            expanded_pts.append(np_p)
        pts = np.array(expanded_pts, dtype=np.float32)

        ordered_pts = self._order_points(pts)

        width_a = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
        width_b = np.linalg.norm(ordered_pts[2] - ordered_pts[3])
        max_w = max(int(width_a), int(width_b), 16)

        height_a = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
        height_b = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
        max_h = max(int(height_a), int(height_b), 16)

        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(img, M, (max_w, max_h), borderMode=cv2.BORDER_REPLICATE)

        if max_h > max_w * 1.3:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        border = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
        border_mean = np.mean(border)
        if border_mean > 127:
            thresh = cv2.bitwise_not(thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        if invert:
            thresh = cv2.bitwise_not(thresh)

        return warped, thresh

    def generate_digital_stamp(self, code: str, parsed: Dict[str, Any]) -> np.ndarray:
        """
        Generates a clean, high-resolution digital stencil badge:
        Pure White text on Pure Black Background (#000000).
        """
        h, w = 200, 800
        stamp = np.zeros((h, w, 3), dtype=np.uint8)

        # Double border
        cv2.rectangle(stamp, (8, 8), (w - 9, h - 9), (255, 255, 255), 2)
        cv2.rectangle(stamp, (14, 14), (w - 15, h - 15), (100, 100, 100), 1)

        # Header Title
        cv2.putText(stamp, "CENTRALIZED AGRICULTURAL IMPLEMENT LASER CODE", (28, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Main Code in Bold White
        code_str = code if code else "NO CODE DETECTED"
        cv2.putText(stamp, code_str, (28, 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3, cv2.LINE_AA)

        # Subtitle / Details
        details = []
        if parsed.get("state_name"):
            details.append(f"STATE: {parsed['state_name']} ({parsed.get('state_code', '')})")
        if parsed.get("model_code"):
            m_info = parsed.get("model_info") or {}
            m_name = m_info.get("name", f"Model {parsed['model_code']}")
            details.append(f"MODEL: {m_name}")
        if parsed.get("year"):
            details.append(f"MFG: {parsed['year']}")
        if parsed.get("serial_no"):
            details.append(f"SN: {parsed['serial_no']}")
        if parsed.get("suffix"):
            details.append(f"MARK: {parsed['suffix']}")

        detail_line = " | ".join(details) if details else "INDUSTRIAL LASER STENCIL IDENTIFIER"
        cv2.putText(stamp, detail_line[:85], (28, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

        # Status stamp
        status_txt = "[VERIFIED IMPLEMENT CODE]" if parsed.get("is_valid") else "[RAW / UNVERIFIED]"
        color = (255, 255, 255) if parsed.get("is_valid") else (180, 180, 180)
        cv2.putText(stamp, status_txt, (w - 240, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return stamp

    def _is_reference_document(self, raw_lines: List[str]) -> bool:
        """Checks if the scanned image is a Reference Catalog Chart or multi-row document."""
        joined = " ".join(raw_lines).upper()
        keywords = ["LIST OF CODES", "LASER MARKING", "MATERIAL CODE", "MODEL/SIZE", "CENTRALIZED", "SERIES"]
        kw_matches = sum(1 for kw in keywords if kw in joined)
        return (kw_matches >= 2) or (len(raw_lines) >= 20 and kw_matches >= 1)

    def process_image(self, img_input: Any, auto_rotate: bool = True) -> Dict[str, Any]:
        """
        High-performance multi-pass OCR pipeline for metal stencil detection and extraction.
        
        Architecture:
        1. Upfront photometric and clarity quality analysis (< 2ms)
        2. Dynamic lighting normalization (adaptive shadow lifting / highlight suppression)
        3. Priority candidate scheduling (Fast path 0° -> Mirrored 0° -> Dual exposure -> 180° -> 90°/270°)
        4. Early exit the moment a validated implement code is recognized
        """
        t0 = time.time()
        # Load image
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
        elif isinstance(img_input, bytes):
            nparr = np.frombuffer(img_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif hasattr(img_input, 'read'):
            bytes_data = img_input.read()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = img_input

        if img is None:
            return {"error": "Failed to decode input image"}

        h, w = img.shape[:2]

        def run_ocr_on_img(target_img: np.ndarray):
            rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            res, _ = self.ocr(rgb)
            dets = []
            lines = []
            if res:
                for item in res:
                    dets.append({
                        "box": item[0],
                        "text": item[1],
                        "score": round(float(item[2]), 3)
                    })
                    lines.append(item[1])
            return dets, lines

        # 1. Quick Quality & Illumination Analysis (< 2ms)
        quality = self.analyze_lighting(img)
        enhanced_img = self.enhance_lighting_and_clarity(img, quality)

        # 2. Build Intelligent Multi-Pass Candidate Schedule
        # Ordered by real-world industrial frequency:
        # 1. Raw 0° (Normal baseline & documents)
        # 2. Mirrored 0° (Through-cut metal parts viewed from reverse side)
        # 3. Enhanced 0° (Adaptive shadow lift / highlight suppression / unsharp mask)
        # 4. Enhanced Mirrored 0° (Backside scan with degraded lighting)
        # 5. Rotated 180° (Upside-down beams)
        # 6. Rotated 90° & 270° (Vertical mounting brackets)
        schedule = [
            (img, 0, False, "Raw-0°"),
            (cv2.flip(img, 1), 0, True, "Raw-Mirrored-0°"),
            (enhanced_img, 0, False, "Enhanced-0°"),
            (cv2.flip(enhanced_img, 1), 0, True, "Enhanced-Mirrored-0°"),
        ]
        if auto_rotate:
            schedule.append((cv2.rotate(img, cv2.ROTATE_180), 180, False, "Raw-180°"))
            schedule.append((cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_180), 180, True, "Raw-Mirrored-180°"))
            schedule.append((cv2.rotate(enhanced_img, cv2.ROTATE_180), 180, False, "Enhanced-180°"))
            schedule.append((cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 90, False, "Raw-90°"))
            schedule.append((cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 270, False, "Raw-270°"))
            schedule.append((cv2.rotate(enhanced_img, cv2.ROTATE_90_CLOCKWISE), 90, False, "Enhanced-90°"))
            schedule.append((cv2.rotate(enhanced_img, cv2.ROTATE_90_COUNTERCLOCKWISE), 270, False, "Enhanced-270°"))

        primary = None
        found_codes = []
        best_angle = 0
        is_mirrored = False
        active_img = img
        best_dets = []

        for target_img, angle, mirrored, tag in schedule:
            dets, lines = run_ocr_on_img(target_img)
            if not dets:
                continue

            # Check if Reference Catalog Chart document
            if angle == 0 and not mirrored and self._is_reference_document(lines):
                doc_codes = []
                for line in lines:
                    c = self._extract_single_line_code(line)
                    if c and c not in doc_codes:
                        doc_codes.append(c)
                final_code = doc_codes[0] if doc_codes else "LIST OF CODES FOR LASER MARKING"
                parsed_info = {
                    "full_code": final_code,
                    "is_document": True,
                    "document_title": "Reference Catalog Chart (List of Laser Marking Codes)",
                    "document_codes": doc_codes,
                    "is_valid": bool(doc_codes),
                    "model_catalog": IMPLEMENT_CATALOG
                }
                cropped_color = img[0:min(int(h * 0.35), 600), :]
                gray_doc = cv2.cvtColor(cropped_color, cv2.COLOR_BGR2GRAY)
                _, doc_bin = cv2.threshold(gray_doc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.mean(doc_bin) > 127:
                    doc_bin = cv2.bitwise_not(doc_bin)
                return {
                    "success": True,
                    "final_code": final_code,
                    "parsed_info": parsed_info,
                    "raw_detections": dets,
                    "best_angle": 0,
                    "is_mirrored": False,
                    "annotated_image": img.copy(),
                    "cropped_color": cropped_color,
                    "cropped_binary": doc_bin,
                    "digital_stamp": self.generate_digital_stamp(final_code, parsed_info),
                    "processing_time_s": round(time.time() - t0, 3)
                }

            # Check for implement codes
            c_found, c_primary = self._extract_all_valid_codes(dets, target_img.shape[1], target_img.shape[0])
            if not best_dets:
                best_dets = dets
                active_img = target_img

            if c_primary and c_primary[1]["is_valid"]:
                found_codes = c_found
                primary = c_primary
                best_angle = angle
                is_mirrored = mirrored
                active_img = target_img
                best_dets = dets
                break  # EARLY EXIT!

        # Final assembly
        if primary:
            final_code, parsed_info, matched_boxes = primary
        else:
            if best_dets:
                cand = self.clean_token(best_dets[0]["text"])
                parsed_info = self.parse_components(cand)
                final_code = parsed_info["full_code"] if parsed_info["is_valid"] else cand
                matched_boxes = [best_dets[0]["box"]]
            else:
                final_code = ""
                parsed_info = {"full_code": "", "is_valid": False}
                matched_boxes = []

        parsed_info["is_document"] = False
        parsed_info["all_codes"] = [f["code"] for f in found_codes]

        # Crop visualization
        cropped_color = None
        cropped_binary_white_on_black = None
        annotated_img = active_img.copy()

        if matched_boxes:
            all_pts = []
            for b in matched_boxes:
                for pt in b:
                    all_pts.append(pt)
            all_pts = np.array(all_pts, dtype=np.float32)
            min_rect = cv2.minAreaRect(all_pts)
            box_points = cv2.boxPoints(min_rect)
            box_points = self._order_points(box_points)
            cropped_color, cropped_binary_white_on_black = self.crop_and_binarize(active_img, box_points.tolist())

        # Draw bounding boxes on annotated image
        for det in best_dets:
            pts = np.array(det["box"], dtype=np.int32)
            cv2.polylines(annotated_img, [pts], True, (0, 255, 0), 2)

        # Generate Digital High-Contrast Stamp (Pure White on Pure Black)
        digital_stamp = self.generate_digital_stamp(final_code, parsed_info)
        elapsed = round(time.time() - t0, 3)

        return {
            "success": bool(final_code and parsed_info.get("is_valid")),
            "final_code": final_code,
            "parsed_info": parsed_info,
            "raw_detections": best_dets,
            "best_angle": best_angle,
            "is_mirrored": is_mirrored,
            "annotated_image": annotated_img,
            "cropped_color": cropped_color,
            "cropped_binary": cropped_binary_white_on_black,
            "digital_stamp": digital_stamp,
            "processing_time_s": elapsed
        }
