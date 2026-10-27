# Laser Cut Metal Stencil OCR & Implement Code Scanner

An industrial vision OCR application and web service built for detecting, deskewing, binarizing, and recognizing laser-cut stencil codes on sheet metal parts, tubes, brackets, and agricultural implements.

🌐 **Live Vercel Production Web App**: [https://raj-ai-ocr.vercel.app](https://raj-ai-ocr.vercel.app)  
📦 **GitHub Repository**: [https://github.com/lostxmusafir/S-ocr.git](https://github.com/lostxmusafir/S-ocr.git)

---

## Deployments & Quick Access

| Interface | URL / Method | Description |
|---|---|---|
| **Vercel Web App (Live)** | [https://raj-ai-ocr.vercel.app](https://raj-ai-ocr.vercel.app) | Responsive dark industrial web UI with camera scan, upload, sample testing, and high-contrast white-on-black cards. |
| **Vercel Cloud API** | `POST https://raj-ai-ocr.vercel.app/api/ocr` | Serverless Python REST API for laser-cut stencil code identification. |
| **Local Streamlit App** | Double-click `run_app.bat` or `http://localhost:8501` | Full interactive desktop dashboard with live camera input and deep analytics. |

---

## Features

- **Multi-Source Input**:
  - 📷 **Live Camera Scanner**: Scan directly via webcam or smartphone camera (supports mobile camera switching & automatic EXIF orientation normalization).
  - 📁 **File Upload**: Upload high-resolution images (PNG, JPG, JPEG, WEBP).
  - 🖼️ **Sample Gallery**: Quick 1-click testing with all 5 sample factory parts.
- **Robust Metal Stencil Recognition & Industrial Edge Enhancement**:
  - Handles arbitrary orientations: horizontal, vertical, 180° upside-down, and skewed parts.
  - **Upfront Photometric Quality Analysis (< 2ms)**: Analyzes mean luminance, contrast standard deviation, shadow clipping, specular glare clipping, and Laplacian edge sharpness.
  - **Dynamic Logarithmic Exposure Normalization**: Automatically lifts underexposed deep shadows in low light and compresses high metallic specular glare.
  - **Mirror / Back-side Scan Detection**: Detects laser cuts photographed from the reverse side of sheet metal plates via horizontal flip evaluation in ~1.4s.
  - **Low Clarity & Blur Edge Restoration**: High-pass unsharp masking sharpens blurred stencil cut contours and bridge boundaries.
  - **Multi-Part Stacked Implement Detection**: Discovers and indexes multiple implement plates stacked on pallets or frames.
  - **Early Exit Fast Path**: High-confidence codes return in ~0.8s without redundant passes.
  - Handles both **1-line** and **2-line** laser stamped layouts regardless of line detection order.
  - Robust against noise text lines, logos, and metal surface scratches.
  - Stencil bridge repair & normalization: handles `4AG`, `AAG`, `4AC`, `4AF`, `AAF`, `NAG`, `M4F`, `4++`, `1A2` mfr bridge, year `202G`/`2024` bridge, `118381` serial bridge split, `UD`/`UIP`/`LIP` suffixes, and tokens with internal spaces.
  - Supports all Indian State prefixes (`GJ`, `MH`, `MP`, `RJ`, `PB`, `HR`, `UP`, `KA`, `TN`, `AP`, `TS`, `DL`, etc.).
- **High-Contrast White-on-Black Visualization ("number white me print or background in black")**:
  - **Identified number printed in glowing White text on pure Black background** (`#000000`).
  - 📋 **1-Click Copy**: Built-in copy-to-clipboard code block for fast data entry.
  - 🔬 **Real Cutout Binarization**: Perspective-deskewed physical cutout patch rendered in high-contrast White text on pure Black.
  - 🏷️ **Digital Label Badge**: Synthetic high-resolution industrial label badge (White on Black) ready for immediate printing or inspection logging.
- **Implement Code Validation & Catalog Recognition**:
  - Validates according to the Indian Centralized Implement Code standard:
    `[State]/[Mfr Code]/[Model Code]/[Year]/[Serial No]/[Suffix]`
  - Semantic breakdown into State (e.g., Gujarat `GJ`), Manufacturer (`142`), Model Series (`444`, `445`, `446`, `447`, `3528`, `3674`), Year, and Serial unit.
  - **Document / Reference Chart Mode**: Intelligently identifies reference catalog documents (like the "List of Codes for Laser Marking"), extracting codes and displaying a searchable table of models, series categories, and sizes.
- **Export & History**:
  - Export detected code (TXT), structured JSON report, high-contrast binary mask (PNG), and digital label badge (PNG).
  - Interactive scan history log with timing metrics.

---

## API Endpoints (Vercel Serverless)

### 1. Health Check
```bash
curl https://raj-ai-ocr.vercel.app/api/health
```

### 2. Recognize Stencil Code from Image
```bash
curl -X POST -F "file=@part_photo.jpg" https://raj-ai-ocr.vercel.app/api/ocr
```

Response JSON:
```json
{
  "status": "success",
  "final_code": "GJ/142/445/2026/43889",
  "parsed_info": {
    "is_valid": true,
    "state_code": "GJ",
    "state_name": "Gujarat",
    "mfr_code": "142",
    "model_code": "445",
    "year": "2026",
    "serial_no": "43889",
    "model_info": {
      "name": "REG V1 / SEMI MSS",
      "series": "Regular / Semi Series",
      "sizes": ["5ft", "5.5ft", "1.55m"]
    }
  },
  "best_angle": 90,
  "elapsed_ms": 1180.5,
  "cropped_binary_base64": "data:image/png;base64,...",
  "digital_stamp_base64": "data:image/png;base64,..."
}
```

---

## Local Development & Testing

### 1. Launch the Streamlit App
Double-click `run_app.bat` or run:
```powershell
& "$env:LOCALAPPDATA\Python\bin\python.exe" -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 2. Run Tests
Verify OCR accuracy across all sample images:
```powershell
& "$env:LOCALAPPDATA\Python\bin\python.exe" test_ocr.py
```

Run automated pytest unit tests:
```powershell
& "$env:LOCALAPPDATA\Python\bin\python.exe" -m pytest test_edge_cases.py -v
```

---

## Pushing to GitHub

All code is staged and committed to the `main` branch. To push to GitHub:
- **Option 1**: Double-click `push_to_github.bat` in the folder.
- **Option 2**: In your terminal, run:
  ```powershell
  git push -u origin main
  ```
