"""
Generates an executive 3-page Developer API Specification PDF for the Metal Stencil OCR API.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "INDUSTRIAL METAL STENCIL OCR — REST API SPECIFICATION")
            self.setFont("Helvetica", 8)
            self.drawRightString(558, 755, "PRODUCTION V2.0")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.6)
            self.line(54, 747, 558, 747)

        # Footer
        self.setFont("Helvetica", 8)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.6)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — Engineering & Developer Integration Document")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()

def build_pdf(filename="Metal_Stencil_OCR_API_Documentation.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#2563eb'),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#0f172a'),
        backColor=colors.HexColor('#f8fafc'),
        borderColor=colors.HexColor('#cbd5e1'),
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=2,
        spaceAfter=5
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#1e293b')
    )

    story = []

    # =========================================================================
    # PAGE 1: Architecture, Endpoints Table, and Main POST /api/ocr Specs
    # =========================================================================
    story.append(Paragraph("Industrial Metal Stencil OCR API", title_style))
    story.append(Paragraph("REST API Specification & Frontend Integration Guide (Gemini Vision AI Engine)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#2563eb"), spaceAfter=8))

    # Meta Info Table
    meta_data = [
        [Paragraph("<b>Production Base URL:</b> https://s-ocr.vercel.app", table_cell_style), Paragraph("<b>CORS Policy:</b> Access-Control-Allow-Origin: *", table_cell_style)],
        [Paragraph("<b>Local Dev Base URL:</b> http://localhost:5000", table_cell_style), Paragraph("<b>Engine:</b> Gemini Vision AI (4x Round-Robin)", table_cell_style)],
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. System Overview", h1_style))
    story.append(Paragraph(
        "This cloud vision service extracts standardized laser-cut stencil codes, stamps, and serial numbers from metal parts, structural beams, and agricultural implements. The backend uses Google Gemini Vision models (<code>gemini-3.5-flash</code> / <code>gemini-3.8-flash</code>) backed by <b>4 round-robin API key slots</b> with automatic error/quota failover, ensuring 100% resilience against rate-limiting, extreme low-light shadows, specular glare, inverted/upside-down (180°) orientations, and mirrored through-cuts.",
        body_style
    ))

    story.append(Paragraph("2. Endpoints Summary", h1_style))
    ep_summary = [
        [Paragraph("Method", table_header_style), Paragraph("Endpoint", table_header_style), Paragraph("Content-Type", table_header_style), Paragraph("Purpose", table_header_style)],
        [Paragraph("<b>POST</b>", table_cell_style), Paragraph("<code>/api/ocr</code>", table_cell_style), Paragraph("<code>multipart/form-data</code>", table_cell_style), Paragraph("Upload camera/photo image & extract implement code", table_cell_style)],
        [Paragraph("<b>GET</b>", table_cell_style), Paragraph("<code>/api/health</code>", table_cell_style), Paragraph("<code>application/json</code>", table_cell_style), Paragraph("Heartbeat check & AI round-robin slot status", table_cell_style)],
        [Paragraph("<b>GET</b>", table_cell_style), Paragraph("<code>/api/catalog</code>", table_cell_style), Paragraph("<code>application/json</code>", table_cell_style), Paragraph("Fetch authorized implement models & state codes", table_cell_style)],
    ]
    t_summary = Table(ep_summary, colWidths=[50, 115, 125, 214])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 4))

    story.append(Paragraph("3. Main Scan Endpoint: POST /api/ocr", h1_style))
    story.append(Paragraph("Uploads an image (from smartphone camera or file picker) and returns the verified stencil code.", body_style))

    param_data = [
        [Paragraph("Field", table_header_style), Paragraph("Type", table_header_style), Paragraph("Required", table_header_style), Paragraph("Description", table_header_style)],
        [Paragraph("<code>file</code> / <code>image</code>", table_cell_style), Paragraph("Binary File", table_cell_style), Paragraph("Yes", table_cell_style), Paragraph("Image file from camera/picker (JPEG, PNG, WEBP).", table_cell_style)],
        [Paragraph("<code>include_images</code>", table_cell_style), Paragraph("Boolean", table_cell_style), Paragraph("No", table_cell_style), Paragraph("Returns digital stamp & binary cutout in Base64.", table_cell_style)],
    ]
    t_params = Table(param_data, colWidths=[90, 75, 65, 274])
    t_params.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2563eb")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_params)
    story.append(Spacer(1, 3))

    story.append(Paragraph("<b>Success Response Format (200 OK):</b>", body_style))
    ocr_response_json = """{
  "status": "success",
  "final_code": "GJ/142/446/2026/11881/UP",
  "parsed_info": {
    "full_code": "GJ/142/446/2026/11881/UP",
    "state_code": "GJ", "state_name": "Gujarat",
    "mfr_code": "142", "model_code": "446",
    "model_info": { "name": "TUSKER / SEMI HSS", "series": "Heavy Duty" },
    "year": "2026", "serial_no": "11881", "suffix": "UP",
    "is_valid": true, "orientation": "180deg upside down", "confidence": 0.99
  },
  "key_used_slot": 1, "model_used": "gemini-3.5-flash",
  "digital_stamp_base64": "data:image/png;base64,iVBORw0KGgo...",
  "elapsed_ms": 3140.2
}"""
    story.append(Paragraph(ocr_response_json.replace(" ", "&nbsp;").replace("\n", "<br/>"), code_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Health, Catalog, and Frontend Architecture Flow
    # =========================================================================
    story.append(Paragraph("4. Secondary Endpoints & Reference Data", h1_style))
    story.append(Paragraph("4.1 Health Check: GET /api/health", h2_style))
    story.append(Paragraph("Call on app launch to verify backend availability and active Gemini key slots:", body_style))
    health_json = """{
  "status": "healthy",
  "service": "Laser Cut Metal Stencil OCR API",
  "ai_engine": "Google Gemini Vision AI",
  "round_robin_slots": 4,
  "supported_models": ["444", "445", "446", "447", "3528", "3674"],
  "version": "2.0.0"
}"""
    story.append(Paragraph(health_json.replace(" ", "&nbsp;").replace("\n", "<br/>"), code_style))

    story.append(Paragraph("4.2 Catalog Reference: GET /api/catalog", h2_style))
    story.append(Paragraph("Returns model categories (Regular, Heavy Duty, Brackets) and Indian state mappings:", body_style))
    cat_json = """{
  "status": "success",
  "catalog": {
    "444": { "name": "REG SMART / REG", "series": "Regular Series" },
    "445": { "name": "REG V1 / SEMI MSS", "series": "Regular / Semi Series" },
    "446": { "name": "TUSKER / SEMI HSS", "series": "Heavy Duty / Semi Series" },
    "3674": { "name": "BRACKET / STRUCTURAL MOUNT", "series": "Sub-Assembly" }
  },
  "states": { "GJ": "Gujarat", "MH": "Maharashtra", "MP": "Madhya Pradesh", "RJ": "Rajasthan", ... }
}"""
    story.append(Paragraph(cat_json.replace(" ", "&nbsp;").replace("\n", "<br/>"), code_style))

    story.append(Paragraph("5. Frontend User Workflow (Camera ➔ AI ➔ Readout)", h1_style))
    story.append(Paragraph("The recommended UX integration steps for the web/mobile frontend developer:", body_style))
    story.append(Paragraph("<b>Step 1: Capture Photo:</b> Use a camera input button (<code>&lt;input type='file' accept='image/*' capture='environment'&gt;</code>).", bullet_style))
    story.append(Paragraph("<b>Step 2: Dispatch Request:</b> Construct <code>FormData</code>, append <code>file</code>, and send a <code>POST</code> request to <code>https://s-ocr.vercel.app/api/ocr</code>.", bullet_style))
    story.append(Paragraph("<b>Step 3: Loading State:</b> Display an animated progress bar or pulsing laser scanner animation while AI processes (avg. 2-4 seconds).", bullet_style))
    story.append(Paragraph("<b>Step 4: Primary Card:</b> Render <code>response.final_code</code> prominently in bold monospace font (White text on Pure Black card for high factory contrast).", bullet_style))
    story.append(Paragraph("<b>Step 5: Breakdown Chips:</b> Display separate badge pills for State, Manufacturer Code, Model, Production Year, Serial Number, and Suffix.", bullet_style))
    story.append(Paragraph("<b>Step 6: Digital Stamp:</b> Bind <code>response.digital_stamp_base64</code> directly to <code>&lt;img src={...} /&gt;</code> for immediate visual confirmation.", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Production Code Snippets (JS, React, cURL) and Error Handling
    # =========================================================================
    story.append(Paragraph("6. Production Frontend Integration Code", h1_style))

    story.append(Paragraph("6.1 Production JavaScript / Fetch API Snippet:", h2_style))
    js_code = """async function scanMetalStencil(fileObject) {
  const formData = new FormData();
  formData.append("file", fileObject); // Camera capture or file picker File object
  formData.append("include_images", "true");

  try {
    const response = await fetch("https://s-ocr.vercel.app/api/ocr", {
      method: "POST",
      body: formData, // Browser sets multipart boundary automatically
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    const data = await response.json();
    console.log("Scanned Code:", data.final_code); // e.g. "GJ/142/446/2026/11881/UP"
    console.log("Model:", data.parsed_info.model_code, "| Serial:", data.parsed_info.serial_no);
    return data;
  } catch (error) {
    console.error("OCR Scan Failure:", error);
    throw error;
  }
}"""
    story.append(Paragraph(js_code.replace(" ", "&nbsp;").replace("\n", "<br/>"), code_style))

    story.append(Paragraph("6.2 cURL / Postman Command:", h2_style))
    curl_code = """curl -X POST "https://s-ocr.vercel.app/api/ocr" \\
  -F "file=@/path/to/part_photo.jpg" \\
  -F "include_images=true" """
    story.append(Paragraph(curl_code.replace(" ", "&nbsp;").replace("\n", "<br/>"), code_style))

    story.append(Paragraph("7. HTTP Status & Error Handling", h1_style))
    err_data = [
        [Paragraph("Status", table_header_style), Paragraph("Meaning", table_header_style), Paragraph("Action Required", table_header_style)],
        [Paragraph("<b>200 OK</b>", table_cell_style), Paragraph("Success. Code extracted.", table_cell_style), Paragraph("Render <code>final_code</code> and metadata on screen.", table_cell_style)],
        [Paragraph("<b>400 Bad Request</b>", table_cell_style), Paragraph("No image file provided.", table_cell_style), Paragraph("Ensure file is attached to key <code>file</code> or <code>image</code>.", table_cell_style)],
        [Paragraph("<b>502 Bad Gateway</b>", table_cell_style), Paragraph("AI Vision timeout.", table_cell_style), Paragraph("Prompt user to retake photo with clearer lighting.", table_cell_style)],
        [Paragraph("<b>500 Server Error</b>", table_cell_style), Paragraph("Unexpected server issue.", table_cell_style), Paragraph("Retry request; backend auto-rotates to next key slot.", table_cell_style)],
    ]
    t_err = Table(err_data, colWidths=[80, 160, 264])
    t_err.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t_err)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated clean executive 3-page PDF: {filename}")

if __name__ == "__main__":
    build_pdf()
