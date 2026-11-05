"""
Laser Cut Metal Stencil OCR & Centralized Implement Code Scanner
Powered by Google Gemini Vision AI (4x Round-Robin API Keys).
Specialized for industrial sheet metal parts, beams, brackets, and tubes.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import json
import io
import time
from datetime import datetime

from gemini_scanner import GeminiRoundRobinEngine, IMPLEMENT_CATALOG, STATE_CODES

# Page configuration
st.set_page_config(
    page_title="Metal Stencil OCR Scanner (Gemini Vision AI)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Industrial High-Contrast Black & White Theme
st.markdown("""
<style>
    /* Dark industrial base styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* High-contrast primary readout card (White on Black) */
    .readout-card {
        background-color: #000000;
        border: 2px solid #333333;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.9), 0 0 16px rgba(255, 255, 255, 0.15);
    }
    
    .readout-title {
        color: #aaaaaa;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .readout-number {
        color: #ffffff !important;
        background-color: #000000 !important;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 36px;
        font-weight: 900;
        letter-spacing: 3px;
        padding: 14px 20px;
        border: 2px dashed #666666;
        border-radius: 8px;
        display: inline-block;
        margin: 12px auto;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.7);
        word-break: break-all;
    }
    
    .status-badge-valid {
        display: inline-block;
        background-color: #0d381e;
        color: #4ade80;
        border: 1px solid #166534;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .status-badge-doc {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .status-badge-pending {
        display: inline-block;
        background-color: #38280d;
        color: #facc15;
        border: 1px solid #854d0e;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .status-badge-key {
        display: inline-block;
        background-color: #2e1065;
        color: #c084fc;
        border: 1px solid #7e22ce;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .component-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    
    .component-label {
        color: #8b949e;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .component-value {
        color: #ffffff;
        font-family: monospace;
        font-size: 18px;
        font-weight: 700;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "gemini_engine" not in st.session_state:
    st.session_state.gemini_engine = GeminiRoundRobinEngine()

engine = st.session_state.gemini_engine

# Header
st.title("⚡ Laser Cut Metal Stencil OCR Scanner")
st.caption("Industrial vision scanner powered by Google Gemini Vision AI with 4x Round-Robin API Keys.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Scanner Settings")

    st.markdown("### 🧠 AI Engine: **Gemini Vision AI**")
    st.caption("Active vision model with multi-key round-robin rotation and automatic failover.")

    with st.expander("🔑 4x Gemini API Keys Status", expanded=True):
        keys = engine.get_keys()
        stats = engine.stats
        for i, k in enumerate(keys):
            masked = f"...{k[-6:]}"
            usage = stats["key_usage"][i] if i < len(stats["key_usage"]) else 0
            st.markdown(f"**Slot #{i+1}**: 🟢 `Active` | `{masked}` | Served: **{usage}**")
        st.caption("Rotation: Automatic Round-Robin across all 4 keys.")

    st.markdown("---")
    input_mode = st.radio(
        "Choose Image Source:",
        ["📁 Upload Image File", "📷 Live Camera Scan", "🖼️ Sample Images Gallery"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("📚 Implement Code Catalog")
    with st.expander("View Supported Implement Models"):
        for m_code, data in IMPLEMENT_CATALOG.items():
            st.markdown(f"**Model `{m_code}`**: {data['name']}")
            st.caption(f"Series: {data['series']} | Sizes: {', '.join(data['sizes'])}")

# Handle Input Image
image_bytes = None
image_name = "scanned_part.jpg"

sample_files = {
    "1. Upside-down Single-Line Beam (GJ/142/446/2026/11881/UP)": "WhatsApp Image 2026-10-22 at 4.55.31 PM.jpeg",
    "2. Two-Line Bracket Part (GJ/142/3674 & /2026/10652)": "WhatsApp Image 2026-10-22 at 4.55.31 PM (1).jpeg",
    "3. Two-Line Metal Plate (GJ/142/3674 & /2026/10568)": "WhatsApp Image 2026-10-22 at 4.55.32 PM.jpeg",
    "4. Vertical Tube Beam (GJ/142/445/2026/43889)": "WhatsApp Image 2026-10-22 at 4.55.32 PM (1).jpeg",
    "5. Reference Catalog Chart (List of Laser Marking Codes)": "WhatsApp Image 2026-10-22 at 4.55.33 PM.jpeg",
    "6. [NEW] Mirrored / Back-side Scan (GJ/142/446/2026/11541)": os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.15 PM.jpeg"),
    "7. [NEW] Harsh Gearbox Shadow & Stencil Bridges (GJ/142/446/2026/12313/UP)": os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.09 PM (1).jpeg"),
    "8. [NEW] Bright Orange Implement Assembly (GJ/142/3674/2026/10800)": os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.16 PM (1).jpeg"),
    "9. [NEW] Multi-Part Stacked Pallet (GJ/142/445/2026/44579)": os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.11 PM (1).jpeg"),
    "10. [NEW] Vertical Beam Macro (GJ/142/445/2026/44579)": os.path.join("new ocr data", "WhatsApp Image 2026-10-27 at 4.02.14 PM (1).jpeg"),
}

if input_mode == "📁 Upload Image File":
    uploaded = st.file_uploader(
        "Upload Sheet Metal Photo (PNG, JPG, JPEG, WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload high-res photo or macro shot of laser-cut stencil text."
    )
    if uploaded is not None:
        image_bytes = uploaded.read()
        image_name = uploaded.name

elif input_mode == "📷 Live Camera Scan":
    st.info("Align the laser-cut numbers inside the camera view and click 'Take Photo'.")
    cam_file = st.camera_input("Capture Metal Part Stencil")
    if cam_file is not None:
        image_bytes = cam_file.read()
        image_name = f"camera_scan_{datetime.now().strftime('%H%M%S')}.jpg"

elif input_mode == "🖼️ Sample Images Gallery":
    selected_sample = st.selectbox("Select a Sample Part Image to Test:", list(sample_files.keys()))
    sample_path = sample_files[selected_sample]
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            image_bytes = f.read()
            image_name = os.path.basename(sample_path)
    else:
        st.error(f"Sample file {sample_path} not found in workspace.")

# Main Processing Section
if image_bytes is not None:
    # Decode image with EXIF orientation correction
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        orig_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        nparr = np.frombuffer(image_bytes, np.uint8)
        orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    col_run, col_time = st.columns([2, 4])
    with col_run:
        scan_button = st.button("🚀 Analyze & Extract Stencil Number", type="primary", use_container_width=True)

    scan_id = f"{image_name}"
    needs_processing = scan_button or (st.session_state.get("last_scan_id") != scan_id)

    if needs_processing:
        start_t = time.time()
        with st.spinner("Analyzing metal stencil with Gemini Vision AI (Round-Robin)..."):
            result = engine.process_image(orig_bgr)
        elapsed_ms = round((time.time() - start_t) * 1000, 1)
        result["elapsed_ms"] = elapsed_ms
        st.session_state.current_result = result
        st.session_state.last_scan_id = scan_id
        
        # Save to scan history (avoid consecutive duplicate entries)
        final_c = result.get("final_code", "")
        if final_c:
            hist = st.session_state.scan_history
            if not hist or hist[0]["code"] != final_c or hist[0]["file"] != image_name:
                st.session_state.scan_history.insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file": image_name,
                    "code": final_c,
                    "is_valid": result.get("parsed_info", {}).get("is_valid", False),
                    "elapsed_ms": elapsed_ms
                })

    result = st.session_state.get("current_result")

    if result:
        st.markdown("---")
        
        final_code = result.get("final_code", "")
        parsed = result.get("parsed_info", {})
        is_valid = parsed.get("is_valid", False)
        is_document = parsed.get("is_document", False)
        elapsed_ms = result.get("elapsed_ms", 0)

        # ---------------------------------------------------------------------
        # 1. PRIMARY RESULT CARD: WHITE TEXT ON PURE BLACK BACKGROUND
        # ---------------------------------------------------------------------
        badges = []
        if is_document:
            badges.append('<span class="status-badge-doc">📄 REFERENCE SPECIFICATION CATALOG SHEET</span>')
            card_title = "MASTER REFERENCE SPECIFICATION SHEET"
        elif is_valid:
            badges.append('<span class="status-badge-valid">✓ VALID CENTRALIZED IMPLEMENT CODE</span>')
            card_title = "IDENTIFIED STENCIL CODE / NUMBER"
        else:
            badges.append('<span class="status-badge-pending">ℹ RAW / UNSTRUCTURED TEXT</span>')
            card_title = "IDENTIFIED TEXT"

        if result.get("key_used_slot"):
            badges.append(f'<span class="status-badge-key">🔑 GEMINI KEY #{result["key_used_slot"]} ({result.get("key_masked", "")})</span>')

        if result.get("model_used"):
            badges.append(f'<span class="status-badge-doc">🤖 {result["model_used"]}</span>')

        if result.get("is_mirrored"):
            badges.append('<span class="status-badge-pending">🪞 BACK-SIDE MIRRORED SCAN</span>')

        b_angle = result.get("best_angle", "0deg")
        if b_angle and b_angle != "0deg" and b_angle != 0:
            badges.append(f'<span class="status-badge-doc">🔄 {b_angle}</span>')

        status_html = " ".join(badges)

        st.markdown(f"""
        <div class="readout-card">
            <div class="readout-title">{card_title}</div>
            <div class="readout-number">{final_code if final_code else "NO CODE DETECTED"}</div>
            <div>
                {status_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick copy code block
        st.caption(f"📋 Click the copy icon in the top-right corner to copy the detected number ({elapsed_ms} ms):")
        st.code(final_code, language="text")

        # Multi-code stacked implements alert
        all_codes = parsed.get("all_codes", [])
        if len(all_codes) > 1:
            st.info(f"**Multi-Part Stacked Implement Alert:** Found {len(all_codes)} distinct implement codes in this frame: " + " | ".join([f"`{c}`" for c in all_codes]))

        # Notes
        if parsed.get("notes"):
            st.caption(f"💡 **AI Vision Observations:** {parsed['notes']}")

        # ---------------------------------------------------------------------
        # 2. HIGH-CONTRAST DIGITAL STENCIL BADGE & CUTOUT (WHITE ON BLACK)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔬 High-Contrast Cutout (White Text on Black Background)")
        
        c_bin1, c_bin2 = st.columns([1, 1])
        
        with c_bin1:
            cropped_bin = result.get("cropped_binary")
            if cropped_bin is not None:
                st.image(
                    cropped_bin,
                    caption="Extracted Stencil Patch (White Characters on Pure Black)",
                    use_container_width=True,
                    clamp=True
                )
            else:
                st.info("No localized stencil patch cropped.")

        with c_bin2:
            digital_stamp = result.get("digital_stamp")
            if digital_stamp is not None:
                stamp_rgb = cv2.cvtColor(digital_stamp, cv2.COLOR_BGR2RGB)
                st.image(
                    stamp_rgb,
                    caption="Generated High-Contrast Digital Label Badge (White on Black)",
                    use_container_width=True
                )

        # ---------------------------------------------------------------------
        # 3. SEMANTIC BREAKDOWN OR DOCUMENT CATALOG TABLE
        # ---------------------------------------------------------------------
        st.markdown("---")
        if is_document:
            st.subheader("📚 Master Centralized Implement Catalog")
            st.write("Identified specifications from the Manufacturer Master Laser Marking Chart:")
            
            catalog_rows = []
            for m_id, m_data in IMPLEMENT_CATALOG.items():
                catalog_rows.append({
                    "Model Code": m_id,
                    "Implement Name": m_data["name"],
                    "Series Category": m_data["series"],
                    "Supported Sizes": ", ".join(m_data["sizes"]),
                    "Standard Marking Code": f"GJ/142/{m_id}/2023/0000"
                })
            st.dataframe(catalog_rows, use_container_width=True)

            doc_codes = parsed.get("document_codes", [])
            if doc_codes:
                st.markdown(f"**Identified Implement Codes in Sheet:** " + ", ".join([f"`{c}`" for c in doc_codes]))

        else:
            st.subheader("📋 Centralized Implement Code Breakdown")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.markdown(f"""
                <div class="component-box">
                    <div class="component-label">State</div>
                    <div class="component-value">{parsed.get('state_code', '-') or '-'}</div>
                    <div style="font-size: 10px; color: #888;">{parsed.get('state_name', '')}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown(f"""
                <div class="component-box">
                    <div class="component-label">Mfr Code</div>
                    <div class="component-value">{parsed.get('mfr_code', '-') or '-'}</div>
                    <div style="font-size: 10px; color: #888;">Authorized Mfr</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c3:
                st.markdown(f"""
                <div class="component-box">
                    <div class="component-label">Model Code</div>
                    <div class="component-value">{parsed.get('model_code', '-') or '-'}</div>
                    <div style="font-size: 10px; color: #888;">{parsed.get('model_info', {}).get('series', '') if parsed.get('model_info') else ''}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c4:
                st.markdown(f"""
                <div class="component-box">
                    <div class="component-label">Mfg Year</div>
                    <div class="component-value">{parsed.get('year', '-') or '-'}</div>
                    <div style="font-size: 10px; color: #888;">Production Year</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c5:
                st.markdown(f"""
                <div class="component-box">
                    <div class="component-label">Serial No</div>
                    <div class="component-value">{parsed.get('serial_no', '-') or '-'}</div>
                    <div style="font-size: 10px; color: #888;">5-Digit Serial</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c6:
                st.markdown(f"""
                <div class="component-box">
                    <div class="component-label">Suffix / Pos</div>
                    <div class="component-value">{parsed.get('suffix', '-') or '-'}</div>
                    <div style="font-size: 10px; color: #888;">Assembly Suffix</div>
                </div>
                """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 4. RAW DETECTIONS & EXPORT
        # ---------------------------------------------------------------------
        st.markdown("---")
        exp_col1, exp_col2 = st.columns([3, 1])
        with exp_col1:
            st.subheader("📊 Scan History & Audit Log")
        with exp_col2:
            if st.session_state.scan_history:
                csv_data = "Timestamp,File,Code,Valid,Elapsed_ms\n"
                for h in st.session_state.scan_history:
                    csv_data += f'"{h["timestamp"]}","{h["file"]}","{h["code"]}",{h["is_valid"]},{h["elapsed_ms"]}\n'
                st.download_button(
                    "📥 Export History (CSV)",
                    data=csv_data,
                    file_name=f"stencil_scans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if st.session_state.scan_history:
            st.dataframe(st.session_state.scan_history, use_container_width=True)
        else:
            st.info("No scans recorded yet. Upload an image or select a sample to begin.")
