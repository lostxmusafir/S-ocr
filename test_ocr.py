"""
Comprehensive test suite for StencilOCREngine across all 18 images
(4 original sample images + 13 new real-world images in 'new ocr data' + 1 reference chart).
"""
import os
import glob
import time
from ocr_engine import StencilOCREngine

def main():
    engine = StencilOCREngine()
    
    expected_matches = {
        # Original 4 implement images
        "WhatsApp Image 2026-10-22 at 4.55.31 PM.jpeg": "GJ/142/446/2026/11881/UP",
        "WhatsApp Image 2026-10-22 at 4.55.31 PM (1).jpeg": "GJ/142/3674/2026/10652",
        "WhatsApp Image 2026-10-22 at 4.55.32 PM.jpeg": "GJ/142/3674/2026/10568",
        "WhatsApp Image 2026-10-22 at 4.55.32 PM (1).jpeg": "GJ/142/445/2026/43889",
        
        # 13 new real-world challenge images
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.09 PM (1).jpeg"): "GJ/142/446/2026/12313/UP",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.09 PM (2).jpeg"): "GJ/142/446/2026/12313/UP",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.09 PM.jpeg"): "GJ/142/446/2026/12313/UP",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.10 PM.jpeg"): "GJ/142/446/2026/12313/UP",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.11 PM (1).jpeg"): "GJ/142/445/2026/44579",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.11 PM.jpeg"): "GJ/142/446/2026/12313/UP",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.14 PM (1).jpeg"): "GJ/142/445/2026/44579",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.14 PM.jpeg"): "GJ/142/445/2026/44579",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.15 PM (1).jpeg"): "GJ/142/446/2026/12733/UP",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.15 PM.jpeg"): "GJ/142/446/2026/11541",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.16 PM (1).jpeg"): "GJ/142/3674/2026/10800",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.16 PM.jpeg"): "GJ/142/446/2026/12733/UP",
        os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.17 PM.jpeg"): "GJ/142/3674/2026/10801",
    }
    
    print("======================================================================")
    print("RUNNING INDUSTRIAL METAL STENCIL OCR VERIFICATION")
    print("======================================================================")
    
    passed = 0
    total = len(expected_matches)
    times = []
    
    for filename, expected in expected_matches.items():
        if not os.path.exists(filename):
            print(f"[FAIL] Missing file: {filename}")
            continue
            
        t0 = time.time()
        result = engine.process_image(filename)
        dt = round(time.time() - t0, 3)
        times.append(dt)
        
        code = result.get("final_code")
        parsed = result.get("parsed_info", {})
        
        is_exact = (code == expected)
        status = "PASSED" if is_exact else "FAILED"
        if is_exact:
            passed += 1
            
        print(f"[{status}] File: {filename} ({dt:.2f}s)")
        print(f"   Expected: {expected}")
        print(f"   Detected: {code}")
        print(f"   Valid Implement Code: {parsed.get('is_valid')} | Angle: {result.get('best_angle', 0)}° | Mirrored: {result.get('is_mirrored', False)}")
        print(f"   Components: State={parsed.get('state_code')}, Mfr={parsed.get('mfr_code')}, Model={parsed.get('model_code')}, Year={parsed.get('year')}, Serial={parsed.get('serial_no')}, Suffix={parsed.get('suffix')}")
        if result.get("cropped_binary") is not None:
            h, w = result["cropped_binary"].shape[:2]
            print(f"   High-Contrast Crop (White on Black): {w}x{h} px")
        if result.get("digital_stamp") is not None:
            sh, sw = result["digital_stamp"].shape[:2]
            print(f"   Digital Stencil Badge (White on Black): {sw}x{sh} px")
        if parsed.get("all_codes") and len(parsed["all_codes"]) > 1:
            print(f"   Additional Detected Codes: {parsed['all_codes'][1:]}")
        print("-" * 70)
        
    print(f"\nFinal Result: {passed}/{total} Stencil Image Tests Passed ({passed/total*100:.1f}%)")
    if times:
        print(f"Average Execution Time: {sum(times)/len(times):.2f}s per image")
    
    # Verify reference chart parsing
    chart_file = "WhatsApp Image 2026-10-22 at 4.55.33 PM.jpeg"
    if os.path.exists(chart_file):
        chart_result = engine.process_image(chart_file, auto_rotate=False)
        parsed_chart = chart_result.get("parsed_info", {})
        print(f"\n[PASSED] Reference Chart File Processed:")
        print(f"   Document Title: {parsed_chart.get('document_title')}")
        print(f"   Is Document: {parsed_chart.get('is_document')}")
        print(f"   Total Text Items Detected: {len(chart_result['raw_detections'])}")
        print(f"   Identified Codes in Sheet: {parsed_chart.get('document_codes')}")
        print("======================================================================")

    assert passed == total, f"Expected all {total} images to pass, but only {passed} passed!"

if __name__ == "__main__":
    main()
