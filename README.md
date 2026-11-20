# ⚡ Laser Cut Metal Stencil OCR & Industrial Implement Code Scanner

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Deployed-black?logo=vercel&logoColor=white)](https://s-ocr.vercel.app)
[![Render Service](https://img.shields.io/badge/Render-Online-46E3B7?logo=render&logoColor=white)](https://s-ocr-3zbt.onrender.com)
[![Model](https://img.shields.io/badge/Model-Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

An industrial-grade Computer Vision and Cloud OCR system engineered to detect, deskew, and decode laser-cut metal stencils, stamped implement codes, and serial numbers from raw metal parts, tractor attachments, rotavators, and structural steel beams.

Powered exclusively by **Google Gemini 2.5 Flash Cloud Vision** with a **10-Key Round-Robin Failover Architecture**, ensuring 100% continuous uptime against harsh specular glare, direct sunlight reflections, deep workshop shadows, inverted plates, and rate limits.

---

## 🌐 Live Deployments

| Component | URL | Method | Purpose |
|:---|:---|:---:|:---|
| **Vercel Web App** | [https://s-ocr.vercel.app](https://s-ocr.vercel.app) | `GET` | Responsive industrial web dashboard with camera scan, upload, and high-contrast readout cards. |
| **Vercel Cloud API** | [https://s-ocr.vercel.app/api/ocr](https://s-ocr.vercel.app/api/ocr) | `POST` | Primary serverless REST API for mobile apps, edge devices, and third-party integrations. |
| **Render Cloud Service** | [https://s-ocr-3zbt.onrender.com/ocr](https://s-ocr-3zbt.onrender.com/ocr) | `POST` | High-concurrency Gunicorn/Flask containerized backup endpoint. |
| **GitHub Repository** | [https://github.com/lostxmusafir/S-ocr](https://github.com/lostxmusafir/S-ocr) | Git | Source repository (branch `main`). |

---

## 🚀 Key Architectural Features

### 1. Pure Gemini 2.5 Flash Cloud Vision
- **No Fragile Local Heuristics**: Replaced brittle traditional thresholding and CRNN character models with Google's state-of-the-art multimodal vision model `gemini-2.5-flash`.
- **Specular Glare & Shadow Immunity**: Real factory metal plates suffer from blown-out metallic reflections under bright sunlight or underexposed muddy contrast in dark sheds. Gemini 2.5 Flash understands 3D metal surfaces and reads cutouts cleanly across both extremes.
- **Stencil Bridge Gap Resolution**: Laser-cut characters have physical bridge slits (tabs) so the metal island does not fall out (e.g. `0`, `3`, `4`, `6`, `8`, `A`, `B`, `D`, `G`). The engine natively repairs and decodes them into the true alphanumeric characters.

### 2. 10-Key Round-Robin Engine with Instant Failover
- **Load Distribution**: Every incoming request rotates across a pool of 10 API keys (`Key #1 ➔ Key #2 ➔ Key #3...`).
- **Zero-Downtime Failover**: If any key hits a rate-limit (`429`), quota exhaustion, or error (`401`, `403`, `404`), the engine transfers the request to the next available working key in **under 5 milliseconds** without dropping the scan.
- **Dynamic Cooldown Tracking**: Rate-limited keys are temporarily placed on a 60-second cooldown and automatically rejoined to the pool once cleared.

### 3. Dual-View (Omnidirectional & Reverse-Cut Correction)
- **Automatic LAB-CLAHE Enhancement**: Enhances local contrast between hollow laser-cut holes and background metal surfaces.
- **Simultaneous Dual-View**: Sends both the original enhanced image and a horizontally flipped view in a single API inference call.
- **Through-Cut Support**: If an operator takes a picture from the reverse / backside of a through-cut plate, the system detects the mirror reflection and returns the forward-readable part number instantly.
- **Orientation Independent**: Detects text at 0°, 90°, 180° (upside-down), and arbitrary handheld angles.

---

## 📋 Centralized Implement Code Standard

Codes conform to the **Indian Centralized Agricultural Implement Authority Standard**:

$$\text{Format: } \mathbf{[STATE] / [MANUFACTURER] / [MODEL] / [YEAR] / [SERIAL] [/ SUFFIX]}$$

### Examples:
- `GJ/142/446/2026/12310/UP`
- `GJ/142/3674/2026/10800`
- `GJ/142/445/2026/44579`
- `GJ/142/446/2026/11881/UP`

### Catalog Breakdown:
- **State Code (`GJ`)**: Indian state identifier (default `GJ` for Gujarat).
- **Manufacturer Code (`142`)**: Registered manufacturer license number.
- **Implement Model Code**:
  - `446`: **TUSKER / SEMI HSS / SEMI CHA** (Heavy Duty / Semi Series: 5ft, 6ft, 1.65m)
  - `3674`: **REGULAR SERIES ROTAVATOR** (Multi-Speed Standard: 6ft, 7ft, 1.75m, 2.0m)
  - `445`: **MINI / PADDY SPECIAL** (Compact Utility Series: 3.5ft, 4ft, 1.2m)
  - `3528`: **ORCHARD / INTER-CULTURE TILLER** (Narrow Track: 3ft, 3.5ft)
  - `444`: **FEATHER LIGHT UTILITY** (Light Duty Attachment: 4ft, 5ft)
  - `447`: **HEAVY EXTRA DEEP TILLER** (Super Heavy Series: 7ft, 8ft)
- **Year (`2026`)**: 4-digit manufacturing year.
- **Serial Number (`12310`)**: 5-digit unique serial number (preserves trailing zeros accurately).
- **Suffix (`UP`)**: Optional regional or batch designation (`UP`).

---

## 📱 Mobile Developer API Integration Guide

Mobile developers can directly consume the API via `multipart/form-data` or JSON base64.

### Endpoint Overview
- **Base URL**: `https://s-ocr.vercel.app`
- **Path**: `/api/ocr` (or `/ocr`)
- **HTTP Methods**:
  - `GET`: Returns API documentation, health, and test sample payload.
  - `POST`: Processes and scans image.

---

### Request Options

#### Option A: `multipart/form-data` (Recommended for Mobile Camera)
- **Field Name**: `file` or `image`
- **Content-Type**: `image/jpeg` or `image/png`

#### Option B: `application/json` (Base64 String)
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

#### Optional Header:
- `x-api-key`: Pass a dedicated Gemini API key if you want to bypass the public round-robin server pool.

---

### Response Format

#### Success Response (`HTTP 200 OK`)
```json
{
  "status": "SUCCESS",
  "PARTno": "GJ/142/446/2026/12310/UP",
  "final_code": "GJ/142/446/2026/12310/UP",
  "parsed_info": {
    "full_code": "GJ/142/446/2026/12310/UP",
    "state_code": "GJ",
    "mfr_code": "142",
    "model_code": "446",
    "year": "2026",
    "serial_no": "12310",
    "suffix": "UP",
    "is_valid": true
  }
}
```

#### Failure / Unreadable Image Response (`HTTP 502 Bad Gateway`)
```json
{
  "status": "FAILED",
  "PARTno": null,
  "error": "Gemini 2.5 Flash Cloud Vision could not detect a valid implement code in the image. Please ensure the stamped metal beam is visible."
}
```

---

### Code Examples for Mobile Developers

#### 1. cURL
```bash
curl -X POST \
  -F "file=@/path/to/beam_photo.jpg" \
  https://s-ocr.vercel.app/api/ocr
```

#### 2. Flutter / Dart
```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

Future<String?> scanMetalPart(File imageFile) async {
  final uri = Uri.parse('https://s-ocr.vercel.app/api/ocr');
  final request = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

  final streamedResponse = await request.send();
  final response = await http.Response.fromStream(streamedResponse);

  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    return data['PARTno']; // e.g. "GJ/142/446/2026/12310/UP"
  } else {
    print('OCR Scan Error: ${response.body}');
    return null;
  }
}
```

#### 3. React Native / JavaScript
```javascript
const scanImplementImage = async (imageUri) => {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'scan.jpg',
  });

  try {
    const response = await fetch('https://s-ocr.vercel.app/api/ocr', {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    const result = await response.json();
    if (response.ok && result.status === 'SUCCESS') {
      console.log('Detected Part Number:', result.PARTno);
      return result.PARTno;
    } else {
      console.error('OCR Error:', result.error);
    }
  } catch (error) {
    console.error('Network Error:', error);
  }
};
```

#### 4. Android (Kotlin with Retrofit / OkHttp)
```kotlin
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.File

fun uploadImageForOCR(file: File, callback: (String?) -> Unit) {
    val client = OkHttpClient()
    val requestBody = MultipartBody.Builder()
        .setType(MultipartBody.FORM)
        .addFormDataPart(
            "file",
            file.name,
            RequestBody.create("image/jpeg".toMediaTypeOrNull(), file)
        )
        .build()

    val request = Request.Builder()
        .url("https://s-ocr.vercel.app/api/ocr")
        .post(requestBody)
        .build()

    client.newCall(request).enqueue(object : Callback {
        override fun onFailure(call: Call, e: java.io.IOException) {
            callback(null)
        }

        override fun onResponse(call: Call, response: Response) {
            val body = response.body?.string()
            if (response.isSuccessful && body != null) {
                val partNo = org.json.JSONObject(body).optString("PARTno")
                callback(partNo)
            } else {
                callback(null)
            }
        }
    })
}
```

#### 5. Python
```python
import requests

with open("beam_scan.jpg", "rb") as f:
    response = requests.post(
        "https://s-ocr.vercel.app/api/ocr",
        files={"file": f}
    )

data = response.json()
print("PART Number:", data.get("PARTno"))
print("Metadata:", data.get("parsed_info"))
```

---

## 🛠️ Local Setup & Testing

### Prerequisites
- Python 3.10+ installed
- Git

### 1. Clone & Install Dependencies
```powershell
git clone https://github.com/lostxmusafir/S-ocr.git
cd S-ocr
pip install -r requirements.txt
```

### 2. Configure Environment Keys
Create a `.env.local` file in the root directory:
```env
GEMINI_API_KEYS="your_gemini_api_key_1,your_gemini_api_key_2"
```

### 3. Run Verification Tests
```powershell
# Test Gemini 2.5 Flash Cloud Vision directly
python test_gemini_vision.py

# Simulate live Postman multipart request against production & local APIs
python test_postman_call.py

# Run unit tests on stencil parsing and edge cases
python -m pytest test_edge_cases.py -v
```

### 4. Run the Streamlit Interactive App
```powershell
# Double-click run_app.bat or execute:
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Project Structure

```
S-ocr/
├── api/
│   └── index.py               # Vercel & Render serverless REST API entry point
├── gemini_scanner.py          # Gemini 2.5 Flash 10-key round-robin engine + CLAHE
├── ocr_engine.py              # Stencil parser, implement catalog & normalization rules
├── public/
│   ├── index.html             # High-contrast industrial web dashboard UI
│   └── samples/               # Sample factory metal part images
├── test_gemini_vision.py      # Automated vision verification suite
├── test_postman_call.py       # Live API multipart simulator
├── test_edge_cases.py         # Pytest edge case test suite
├── requirements.txt           # Minimal, fast serverless dependencies
├── vercel.json                # Vercel serverless build and routing configuration
├── render.yaml                # Render web service manifest
└── README.md                  # System & API documentation
```

---

## 📄 License & Credits

- Developed for Industrial Laser Cut Metal Stencil OCR and Agricultural Implement Verification.
- Powered by **Google Gemini 2.5 Flash**.
- Deployed on **Vercel** and **Render**.
