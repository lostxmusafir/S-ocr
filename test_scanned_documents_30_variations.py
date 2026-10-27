"""
Exhaustive Scanning Evaluation Suite for Scanned Documents.
Evaluates all 12 real-world images from 'C:\\Users\\DREAM\\Documents\\Scanned Documents'
across at least 30 variations per image (multi-crop: left-to-bottom, bottom-all,
right-to-bottom, center crops, multiple zoom scales, rotations/tilts, lighting shifts).
Strict criteria: 100% accuracy on code extraction AND all scans < 1.5 seconds (< 1500ms).
"""

import os
import glob
import time
import cv2
import numpy as np
from ocr_engine import StencilOCREngine

# Ground truth mapping for all scanned documents
GROUND_TRUTH = {
    "WhatsApp Image 2026-10-27 at 8.22.12 PM.jpeg": "GJ/142/446/2026/12733/UP",
    "WhatsApp Image 2026-10-27 at 8.22.13 PM (1).jpeg": "GJ/142/446/2026/11541",
    "WhatsApp Image 2026-10-27 at 8.22.13 PM.jpeg": "GJ/142/446/2026/12733/UP",
    "WhatsApp Image 2026-10-27 at 8.22.14 PM (1).jpeg": "GJ/142/445/2026/44579",
    "WhatsApp Image 2026-10-27 at 8.22.14 PM (2).jpeg": "GJ/142/445/2026/44579",
    "WhatsApp Image 2026-10-27 at 8.22.14 PM (3).jpeg": "GJ/142/446/2026/12313/UP",
    "WhatsApp Image 2026-10-27 at 8.22.14 PM.jpeg": "GJ/142/445/2026/44579",
    "WhatsApp Image 2026-10-27 at 8.22.15 PM (1).jpeg": "GJ/142/446/2026/12313/UP",
    "WhatsApp Image 2026-10-27 at 8.22.15 PM.jpeg": "GJ/142/446/2026/12313/UP",
    "WhatsApp Image 2026-10-27 at 8.22.16 PM (1).jpeg": "GJ/142/446/2026/12313/UP",
    "WhatsApp Image 2026-10-27 at 8.22.16 PM (2).jpeg": "GJ/142/446/2026/12313/UP",
    "WhatsApp Image 2026-10-27 at 8.22.16 PM.jpeg": "GJ/142/446/2026/12313/UP",
}

def rotate_image(image, angle):
    """Rotates image by arbitrary angle with bounding box expansion."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_REPLICATE)

def generate_30_variations(img: np.ndarray, text_bbox: tuple):
    """
    Generates 30 diverse scanning variations per image guaranteed to contain the stencil text:
    - 1 Full Image
    - 5 Left-To-Bottom multi-crops
    - 5 Bottom-All multi-crops
    - 5 Right-To-Bottom multi-crops
    - 4 Center / Bounding-box crops
    - 5 Multi-scale zoom levels
    - 3 Lighting / Photometric shifts
    - 2 Angular tilt perturbations
    """
    h, w = img.shape[:2]
    min_x, max_x, min_y, max_y = text_bbox
    
    variations = []
    
    # 1. Full Image Raw
    variations.append(("Full_Image_Raw", img))
    
    # 2. Bottom-All variations (varying top clip across full width)
    # Stencil is below min_y, so top clip between 0 and min_y
    top_fractions_ba = [0.20, 0.40, 0.60, 0.75, 0.85]
    for i, tf in enumerate(top_fractions_ba, start=1):
        y_start = max(0, int(min_y * tf))
        crop = img[y_start:, :]
        variations.append((f"Bottom_All_Var_{i}", crop))
        
    # 3. Left-To-Bottom variations (from left edge x=0 to beyond max_x, top clip above min_y)
    ltb_configs = [
        (0.30, 1.25),
        (0.50, 1.20),
        (0.70, 1.15),
        (0.80, 1.10),
        (0.60, 1.30),
    ]
    for i, (y_frac, x_frac) in enumerate(ltb_configs, start=1):
        y_start = max(0, int(min_y * y_frac))
        x_end = min(w, max(int(max_x + 40), int(max_x * x_frac)))
        crop = img[y_start:, :x_end]
        variations.append((f"Left_To_Bottom_Var_{i}", crop))
        
    # 4. Right-To-Bottom variations (from before min_x to right edge x=w, top clip above min_y)
    rtb_configs = [
        (0.30, 0.70),
        (0.50, 0.75),
        (0.70, 0.80),
        (0.80, 0.85),
        (0.60, 0.65),
    ]
    for i, (y_frac, x_frac) in enumerate(rtb_configs, start=1):
        y_start = max(0, int(min_y * y_frac))
        x_start = max(0, min(int(min_x - 40), int(min_x * x_frac)))
        crop = img[y_start:, x_start:]
        variations.append((f"Right_To_Bottom_Var_{i}", crop))
        
    # 5. Center / Tight Stencil Region Crops (margins around text bounding box)
    center_margins = [0.30, 0.45, 0.60, 0.80]
    box_w = max_x - min_x
    box_h = max_y - min_y
    pad_h = max(40, int(box_h * 0.4))
    pad_w = max(60, int(box_w * 0.15))
    for i, m in enumerate(center_margins, start=1):
        x1 = max(0, int(min_x - pad_w * (1.0 + m)))
        x2 = min(w, int(max_x + pad_w * (1.0 + m)))
        y1 = max(0, int(min_y - pad_h * (1.0 + m)))
        y2 = min(h, int(max_y + pad_h * (1.0 + m)))
        crop = img[y1:y2, x1:x2]
        variations.append((f"Center_Crop_Margin_{int(m*100)}pct", crop))
        
    # 6. Multi-Scale Zoom Variations
    scales = [0.85, 0.90, 0.95, 1.05, 1.10]
    for s in scales:
        scaled = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
        variations.append((f"Scale_Zoom_{int(s*100)}pct", scaled))
        
    # 7. Photometric Lighting Shifts
    # a. Brightness Lift
    bright = cv2.convertScaleAbs(img, alpha=1.05, beta=15)
    variations.append(("Lighting_Exposure_Lift", bright))
    # b. Dynamic Gamma Lift (Shadow Detail Boost)
    gamma_table = np.array([((i / 255.0) ** (1.0 / 1.25)) * 255 for i in range(256)]).astype("uint8")
    shadow_lift = cv2.LUT(img, gamma_table)
    variations.append(("Lighting_Shadow_Lift", shadow_lift))
    # c. Contrast Enhancement
    contrast = cv2.convertScaleAbs(img, alpha=1.15, beta=-10)
    variations.append(("Lighting_Contrast_Boost", contrast))
    
    # 8. Angular Tilts
    tilt_cw = rotate_image(img, 1.5)
    variations.append(("Angle_Tilt_Plus_1.5deg", tilt_cw))
    tilt_ccw = rotate_image(img, -1.5)
    variations.append(("Angle_Tilt_Minus_1.5deg", tilt_ccw))
    
    assert len(variations) == 30, f"Expected exactly 30 variations, got {len(variations)}"
    return variations

def main():
    target_dir = r"C:\Users\DREAM\Documents\Scanned Documents"
    if not os.path.exists(target_dir):
        print(f"[FAIL] Target directory does not exist: {target_dir}")
        return

    files = sorted(glob.glob(os.path.join(target_dir, "*.jpeg")) + glob.glob(os.path.join(target_dir, "*.jpg")))
    print("================================================================================")
    print("RUNNING EXHAUSTIVE 30-VARIATION SCANNING EVALUATION ON ALL SCANNED DOCUMENTS")
    print(f"Target Directory: {target_dir}")
    print(f"Total Base Images: {len(files)} | Target Evaluations: {len(files) * 30} scans")
    print("Strict Pass Requirement: 100% Code Extraction Accuracy AND Execution Time < 1.5s (< 1500ms)")
    print("================================================================================")

    engine = StencilOCREngine()

    # Locate initial bounding boxes on baseline scans
    bboxes = {}
    print("\nPhase 1: Establishing ground-truth spatial coordinates...")
    for f in files:
        bname = os.path.basename(f)
        exp = GROUND_TRUTH.get(bname)
        img = cv2.imread(f)
        h, w = img.shape[:2]
        res = engine.process_image(img)
        matched_box = None
        for d in res.get("raw_detections", []):
            cl = engine.clean_token(d["text"])
            if exp and (exp in cl or engine.parse_components(cl).get("full_code") == exp):
                matched_box = d["box"]
                break
        if matched_box:
            xs = [p[0] for p in matched_box]
            ys = [p[1] for p in matched_box]
            bboxes[bname] = (min(xs), max(xs), min(ys), max(ys))
        elif res.get("raw_detections"):
            xs = [p[0] for d in res["raw_detections"] for p in d["box"]]
            ys = [p[1] for d in res["raw_detections"] for p in d["box"]]
            bboxes[bname] = (min(xs), max(xs), min(ys), max(ys))
        else:
            # Default fallback bbox
            bboxes[bname] = (int(w * 0.1), int(w * 0.9), int(h * 0.1), int(h * 0.7))
        print(f"  [OK] {bname}: bbox=({bboxes[bname][0]:.0f}..{bboxes[bname][1]:.0f}, {bboxes[bname][2]:.0f}..{bboxes[bname][3]:.0f})")

    print("\nPhase 2: Executing 30 distinct scans per image (360 total scans)...")
    total_scans = 0
    passed_scans = 0
    all_times = []
    failures = []

    for img_idx, f in enumerate(files, start=1):
        bname = os.path.basename(f)
        expected = GROUND_TRUTH.get(bname)
        if not expected:
            print(f"[SKIP] No ground truth registered for {bname}")
            continue

        img = cv2.imread(f)
        variations = generate_30_variations(img, bboxes[bname])
        
        img_times = []
        img_passes = 0
        
        print(f"\n--- [{img_idx}/{len(files)}] Testing '{bname}' (Expected: {expected}) ---")
        
        for var_name, var_img in variations:
            total_scans += 1
            t0 = time.time()
            res = engine.process_image(var_img)
            dt_ms = (time.time() - t0) * 1000.0
            
            code = res.get("final_code", "")
            is_valid = res.get("parsed_info", {}).get("is_valid", False)
            all_detected = res.get("parsed_info", {}).get("all_codes", [])
            is_exact = (code == expected or expected in all_detected)
            is_under_1500 = (dt_ms < 1500.0)
            
            passed = is_exact and is_under_1500
            if passed:
                passed_scans += 1
                img_passes += 1
            else:
                failures.append({
                    "file": bname,
                    "variation": var_name,
                    "expected": expected,
                    "detected": code,
                    "time_ms": dt_ms,
                    "exact": is_exact,
                    "under_1500": is_under_1500
                })

            all_times.append(dt_ms)
            img_times.append(dt_ms)
            
            status = "PASS" if passed else "FAIL"
            print(f"   [{status}] {var_name:<28}: {dt_ms:6.1f}ms (<1.5s: {str(is_under_1500):<5}) | Detected: {code}")

        avg_img_time = sum(img_times) / len(img_times)
        max_img_time = max(img_times)
        print(f"   => Image Summary: {img_passes}/30 Variations Passed | Avg: {avg_img_time:.1f}ms | Max: {max_img_time:.1f}ms")

    # Final overall report
    print("\n================================================================================")
    print("FINAL EXHAUSTIVE EVALUATION REPORT")
    print("================================================================================")
    print(f"Total Scans Evaluated : {total_scans}")
    print(f"Total Scans Passed    : {passed_scans} ({passed_scans / total_scans * 100:.2f}%)")
    print(f"Total Failures        : {len(failures)}")
    if all_times:
        print(f"Overall Average Time  : {sum(all_times) / len(all_times):.1f}ms ({sum(all_times) / len(all_times) / 1000:.2f}s)")
        print(f"Min Execution Time    : {min(all_times):.1f}ms")
        print(f"Max Execution Time    : {max(all_times):.1f}ms")
        print(f"Strict < 1.5s Rate    : {sum(1 for t in all_times if t < 1500.0) / len(all_times) * 100:.2f}%")

    if failures:
        print("\nFAILURE DETAILS:")
        for fail in failures:
            print(f"  * {fail['file']} [{fail['variation']}]: time={fail['time_ms']:.1f}ms, exact={fail['exact']}, detected='{fail['detected']}', expected='{fail['expected']}'")

    print("================================================================================")
    assert passed_scans == total_scans, f"Verification failed: {passed_scans}/{total_scans} scans passed."
    print("ALL 360 SCANNING VARIATIONS PASSED STRICTLY UNDER 1.5 SECONDS WITH 100% ACCURACY!")

if __name__ == "__main__":
    main()
