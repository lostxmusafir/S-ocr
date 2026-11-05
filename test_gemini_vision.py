"""
Verification script for Gemini Round-Robin Vision OCR Scanner.
"""
import os
import sys
import time
from gemini_scanner import GeminiRoundRobinEngine

def main():
    engine = GeminiRoundRobinEngine()
    if not engine.get_keys():
        print("No GEMINI_API_KEYS found. Pass GEMINI_API_KEYS env variable to test.")
        return

    test_files = [
        ("WhatsApp Image 2026-10-22 at 4.55.31 PM.jpeg", "GJ/142/446/2026/11881/UP"),
        ("WhatsApp Image 2026-10-22 at 4.55.31 PM (1).jpeg", "GJ/142/3674/2026/10652"),
    ]

    print("=" * 70)
    print("TESTING GEMINI VISION OCR (Keys configured:", len(engine.get_keys()), ")")
    print("=" * 70)

    for path, expected in test_files:
        if not os.path.exists(path):
            continue
        t0 = time.time()
        res = engine.process_image(path)
        dt = round(time.time() - t0, 2)
        code = res.get("final_code")
        slot = res.get("key_used_slot")
        model = res.get("model_used")
        match = "MATCH" if code == expected else "MISMATCH"
        print(f"[{match}] File: {os.path.basename(path)}")
        print(f"   Expected : {expected}")
        print(f"   Detected : {code}")
        print(f"   Key Slot : #{slot} | Model: {model} | Time: {dt}s")
        print("-" * 70)

if __name__ == "__main__":
    main()
