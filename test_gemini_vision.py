"""
Verification script for Gemini Round-Robin Vision OCR Scanner.
Tests across sample metal parts with rotation, inversion, and multiple keys.
"""
import os
import time
from gemini_scanner import GeminiRoundRobinEngine

def main():
    engine = GeminiRoundRobinEngine()
    test_files = [
        ("WhatsApp Image 2026-10-22 at 4.55.31 PM.jpeg", "GJ/142/446/2026/11881/UP"),
        ("WhatsApp Image 2026-10-22 at 4.55.31 PM (1).jpeg", "GJ/142/3674/2026/10652"),
        ("WhatsApp Image 2026-10-22 at 4.55.32 PM.jpeg", "GJ/142/3674/2026/10568"),
        (os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.09 PM (1).jpeg"), "GJ/142/446/2026/12313/UP"),
    ]

    print("=" * 70)
    print("TESTING GEMINI VISION 4X ROUND-ROBIN OCR")
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

    print("\nRound-Robin Key Usage Stats:")
    for i, count in enumerate(engine.stats["key_usage"]):
        print(f"   Slot #{i+1}: {count} requests served")
    print("=" * 70)

if __name__ == "__main__":
    main()
