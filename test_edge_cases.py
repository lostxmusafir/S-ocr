"""
Unit tests and edge-case testing for StencilOCREngine
"""
import pytest
import numpy as np
import cv2
import os
from ocr_engine import StencilOCREngine, IMPLEMENT_CATALOG, STATE_CODES

@pytest.fixture(scope="module")
def engine():
    return StencilOCREngine()

def test_clean_token(engine):
    assert engine.clean_token("CJ/142") == "GJ/142"
    assert engine.clean_token("Cl/142") == "GJ/142"
    assert engine.clean_token("GU/142") == "GJ/142"
    assert engine.clean_token("  GJ/445  ") == "GJ/445"
    assert engine.clean_token("!GJ/142.") == "GJ/142"
    assert engine.clean_token("2O26") == "2026"
    assert engine.clean_token("2O23") == "2023"

def test_parse_components(engine):
    # Standard 5-part code
    parsed = engine.parse_components("GJ/142/446/2026/11881")
    assert parsed["is_valid"] is True
    assert parsed["state_code"] == "GJ"
    assert parsed["state_name"] == "Gujarat"
    assert parsed["mfr_code"] == "142"
    assert parsed["model_code"] == "446"
    assert parsed["year"] == "2026"
    assert parsed["serial_no"] == "11881"
    assert parsed["suffix"] == ""
    assert parsed["model_info"]["series"] == "Heavy Duty / Semi Series"

    # With suffix
    parsed_up = engine.parse_components("GJ/142/446/2026/11881/UP")
    assert parsed_up["is_valid"] is True
    assert parsed_up["suffix"] == "UP"

    # Different state code (e.g. Maharashtra, Rajasthan)
    parsed_mh = engine.parse_components("MH/105/444/2025/12345")
    assert parsed_mh["is_valid"] is True
    assert parsed_mh["state_code"] == "MH"
    assert parsed_mh["state_name"] == "Maharashtra"

    # Invalid / empty
    parsed_invalid = engine.parse_components("RANDOM_TEXT_HERE")
    assert parsed_invalid["is_valid"] is False

def test_normalize_two_line_stencil(engine):
    # Image 2 case
    code, info = engine.normalize_code_from_lines(["CJ/142/3674", "12026/10652"])
    assert code == "GJ/142/3674/2026/10652"
    assert info["is_valid"] is True

    # Image 3 case: slashes read as 1
    code, info = engine.normalize_code_from_lines(["GJ/14213674", "12026110568"])
    assert code == "GJ/142/3674/2026/10568"
    assert info["is_valid"] is True

    # Reversed order: line 2 before line 1
    code_rev, info_rev = engine.normalize_code_from_lines(["12026110568", "GJ/14213674"])
    assert code_rev == "GJ/142/3674/2026/10568"
    assert info_rev["is_valid"] is True

    # With noise lines
    code_noise, info_noise = engine.normalize_code_from_lines([
        "FACTORY LOGO", "12026/10652", "SECTION 4", "CJ/142/3674", "WELD 01"
    ])
    assert code_noise == "GJ/142/3674/2026/10652"
    assert info_noise["is_valid"] is True

def test_normalize_single_line_stencil(engine):
    # Image 1 case
    code, info = engine.normalize_code_from_lines(["GJ/142/446/2026/11881/UP"])
    assert code == "GJ/142/446/2026/11881/UP"
    assert info["is_valid"] is True

    # Image 4 case
    code, info = engine.normalize_code_from_lines(["GJ/142/445/2026/43889"])
    assert code == "GJ/142/445/2026/43889"
    assert info["is_valid"] is True

    # Embedded code within noise text
    code_emb, info_emb = engine.normalize_code_from_lines(["PART GJ/142/445/2026/43889 LOT 3"])
    assert code_emb == "GJ/142/445/2026/43889"
    assert info_emb["is_valid"] is True

def test_order_points_no_collapse(engine):
    # Rotated quadrilateral at ~45 degrees
    rotated_pts = np.array([[100, 100], [200, 200], [150, 250], [50, 150]], dtype=np.float32)
    ordered = engine._order_points(rotated_pts)
    assert len(ordered) == 4
    # Ensure all 4 vertices are unique (no duplicates)
    unique_rows = np.unique(ordered, axis=0)
    assert len(unique_rows) == 4

def test_digital_stamp_generation(engine):
    parsed = engine.parse_components("GJ/142/446/2026/11881/UP")
    stamp = engine.generate_digital_stamp("GJ/142/446/2026/11881/UP", parsed)
    assert stamp is not None
    assert stamp.shape == (200, 800, 3)
    # Background should be predominantly black
    assert np.mean(stamp[0:5, 0:5]) == 0

def test_reference_document_detection(engine):
    chart_lines = [
        "LIST OF CODES FOR LASER MARKING",
        "SERIES",
        "MATERIAL CODE",
        "MODEL/SIZE",
        "CENTRALIZED IMPLE CODE SIZE",
        "SSUS", "TA147522LM", "7ft SSUS", "GJ/142",
        "REG SMART", "GJ/142/444/2023"
    ]
    assert engine._is_reference_document(chart_lines) is True
    assert engine._is_reference_document(["GJ/142/445/2026/43889"]) is False

def test_empty_and_corrupt_inputs(engine):
    # Blank image
    blank = np.zeros((200, 400, 3), dtype=np.uint8)
    res = engine.process_image(blank, auto_rotate=False)
    assert res["success"] is False or res["final_code"] == ""

    # None input
    res_none = engine.process_image(None)
    assert "error" in res_none

    # Corrupt bytes
    res_corrupt = engine.process_image(b"not_an_image")
    assert "error" in res_corrupt

def test_crop_and_binarize(engine):
    # Create synthetic test image with text
    test_img = np.zeros((100, 300, 3), dtype=np.uint8)
    box = [[10.0, 10.0], [290.0, 10.0], [290.0, 90.0], [10.0, 90.0]]
    cv2.putText(test_img, "GJ/142", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    warped, thresh = engine.crop_and_binarize(test_img, box)
    assert warped is not None
    assert thresh is not None
    # Check that thresh has 0 and 255 values
    assert np.all(np.isin(thresh, [0, 255]))

def test_real_dataset_images(engine):
    samples = {
        "WhatsApp Image 2026-10-22 at 4.55.31 PM.jpeg": "GJ/142/446/2026/11881/UP",
        "WhatsApp Image 2026-10-22 at 4.55.31 PM (1).jpeg": "GJ/142/3674/2026/10652",
        "WhatsApp Image 2026-10-22 at 4.55.32 PM.jpeg": "GJ/142/3674/2026/10568",
        "WhatsApp Image 2026-10-22 at 4.55.32 PM (1).jpeg": "GJ/142/445/2026/43889",
    }
    for filepath, expected in samples.items():
        if os.path.exists(filepath):
            res = engine.process_image(filepath)
            assert res["final_code"] == expected, f"Failed on {filepath}: got {res['final_code']}, expected {expected}"
            assert res["cropped_binary"] is not None
            assert res["digital_stamp"] is not None

def test_stencil_bridge_normalization(engine):
    # Stencil bridge cut gaps in model code (44G, IA4G, IN4G -> 446)
    code1, _ = engine.normalize_code_from_lines(["GJ/142/44G/2026/12313/UP"])
    assert code1 == "GJ/142/446/2026/12313/UP"

    code2, _ = engine.normalize_code_from_lines(["GJ/142IA4G/2026/123137UP"])
    assert code2 == "GJ/142/446/2026/12313/UP"

    code3, _ = engine.normalize_code_from_lines(["GJ/142IN4GI2026I12313/LIP"])
    assert code3 == "GJ/142/446/2026/12313/UP"

    # Year normalization (202G -> 2026)
    code4, _ = engine.normalize_code_from_lines(["GJ/142I44G/202G/12313/UP"])
    assert code4 == "GJ/142/446/2026/12313/UP"

    # Serial trailing digit preservation with suffix (11881/UP)
    code5, _ = engine.normalize_code_from_lines(["GJ/142/446/2026/11881/UP"])
    assert code5 == "GJ/142/446/2026/11881/UP"

def test_mirrored_backside_detection(engine):
    mirror_file = os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.15 PM.jpeg")
    if os.path.exists(mirror_file):
        res = engine.process_image(mirror_file)
        assert res["final_code"] == "GJ/142/446/2026/11541"
        assert res["is_mirrored"] is True
        assert res["parsed_info"]["is_valid"] is True

def test_digital_stamp_none_model_info(engine):
    # Ensure no AttributeError when model_info is None
    parsed = {
        "full_code": "GJ/142/9999/2026/12345",
        "state_code": "GJ",
        "state_name": "Gujarat",
        "mfr_code": "142",
        "model_code": "9999",
        "model_info": None,
        "year": "2026",
        "serial_no": "12345",
        "suffix": "",
        "is_valid": True
    }
    stamp = engine.generate_digital_stamp("GJ/142/9999/2026/12345", parsed)
    assert stamp is not None
    assert stamp.shape == (200, 800, 3)

def test_all_new_dataset_samples(engine):
    new_samples = {
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
    for filepath, expected in new_samples.items():
        if os.path.exists(filepath):
            res = engine.process_image(filepath)
            assert res["final_code"] == expected, f"Failed on {filepath}: got {res['final_code']}, expected {expected}"
            assert res["parsed_info"]["is_valid"] is True

def test_tokens_with_spaces_and_stray_symbols(engine):
    # Spaces around slashes and intra-character gaps
    code1, info1 = engine.normalize_code_from_lines(["GJ / 142 / 446 / 2026 / 12313 / UP"])
    assert code1 == "GJ/142/446/2026/12313/UP"
    assert info1["is_valid"] is True

    # Stray plus and bridged chars
    code2, info2 = engine.normalize_code_from_lines(["GJ/14+2/4+4+/202/12733/UP"])
    assert code2 == "GJ/142/446/2026/12733/UP"
    assert info2["is_valid"] is True

def test_serial_ghost_digit_recovery(engine):
    # 6-digit serial with split 8 ('38' or '83')
    assert engine._normalize_serial("118381") == "11881"
    assert engine._normalize_serial("118831") == "11881"
    # Leading 1 from slash misread
    assert engine._normalize_serial("112313") == "12313"
    # Normal 5-digit serial
    assert engine._normalize_serial("44579") == "44579"

def test_manufacturer_bridge_recovery(engine):
    # Bridged 4 in mfr code (1A2 -> 142)
    code, info = engine.normalize_code_from_lines(["GJ/1A2/AA.G/202/12313/UP"])
    assert code == "GJ/142/446/2026/12313/UP"
    assert info["mfr_code"] == "142"

def test_model_bridge_aliases(engine):
    assert engine._normalize_model("AAG") == "446"
    assert engine._normalize_model("4AC") == "446"
    assert engine._normalize_model("4AF") == "446"
    assert engine._normalize_model("AAF") == "446"
    assert engine._normalize_model("NAG") == "446"
    assert engine._normalize_model("M4F") == "446"
    assert engine._normalize_model("4++") == "446"
    assert engine._normalize_model("44") == "446"

def test_photometric_quality_analysis(engine):
    # Black/dark image
    dark_img = np.zeros((200, 400, 3), dtype=np.uint8)
    q_dark = engine.analyze_lighting(dark_img)
    assert q_dark["is_low_light"] is True
    assert q_dark["needs_enhancement"] is True

    # Overexposed/glare image
    glare_img = np.full((200, 400, 3), 245, dtype=np.uint8)
    q_glare = engine.analyze_lighting(glare_img)
    assert q_glare["is_glare"] is True
    assert q_glare["needs_enhancement"] is True

def test_stress_conditions_resilience(engine):
    # Test sample with low-light, harsh glare, and blur
    img_path = "WhatsApp Image 2026-10-22 at 4.55.31 PM.jpeg"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        expected = "GJ/142/446/2026/11881/UP"

        # 1. Low light (alpha=0.35, beta=-10)
        dark = cv2.convertScaleAbs(img, alpha=0.35, beta=-10)
        res_dark = engine.process_image(dark)
        assert res_dark["final_code"] == expected, f"Dark failed: got {res_dark['final_code']}"

        # 2. Harsh Glare (alpha=1.65, beta=40)
        bright = cv2.convertScaleAbs(img, alpha=1.65, beta=40)
        res_bright = engine.process_image(bright)
        assert res_bright["final_code"] == expected, f"Glare failed: got {res_bright['final_code']}"

        # 3. Blur (Gaussian blur 5x5)
        blur = cv2.GaussianBlur(img, (5, 5), 1.2)
        res_blur = engine.process_image(blur)
        assert res_blur["final_code"] == expected, f"Blur failed: got {res_blur['final_code']}"


