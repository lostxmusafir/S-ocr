"""
Generates an executive, publication-grade Architecture & Interactive Flowchart PDF
for the Industrial Metal Stencil Laser OCR Engine.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Group

class FlowchartCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(FlowchartCanvas, self).__init__(*args, **kwargs)
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
        # Top banner line
        self.setStrokeColor(colors.HexColor("#2563eb"))
        self.setLineWidth(2)
        self.line(40, 755, 572, 755)

        # Header text
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0f172a"))
        self.drawString(40, 762, "INDUSTRIAL METAL STENCIL OCR — SYSTEM ARCHITECTURE & FLOWCHART")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(572, 762, "PRODUCTION V2.5")

        # Footer line
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.8)
        self.line(40, 42, 572, 42)

        # Footer text
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(40, 30, "Confidential — Engineering Architecture & Integration Specification")
        self.drawRightString(572, 30, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf():
    pdf_filename = "Metal_Stencil_OCR_Architecture_Flowchart.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a")
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569")
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6
    )

    step_title = ParagraphStyle(
        "StepTitle",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0f172a")
    )

    step_desc = ParagraphStyle(
        "StepDesc",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    code_style = ParagraphStyle(
        "CodeStyle",
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # Title Banner
    story.append(Paragraph("System Architecture & Processing Flowchart", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Complete technical pipeline diagram for laser-cut metal stencils with multi-angle recovery, "
        "CLAHE contrast isolation, 6-key round-robin AI engine, and automatic catalog validation.",
        subtitle_style
    ))
    story.append(Spacer(1, 12))

    # Flowchart Visual Cards with Arrows
    flow_steps = [
        (
            "STEP 1: IMAGE CAPTURE & ARBITRARY ORIENTATIONS",
            "#2563eb",
            "<b>Input Sources:</b> Mobile App Camera, Handheld Terminal, or API Client.<br/>"
            "<b>Physical Realities:</b> Arbitrary phone rotation (0°, 90°, 180°, 270°), mirrored reverse-side shots, "
            "laser stencil bridge gaps, harsh specular metal glare, and background floor colors showing through slits."
        ),
        (
            "STEP 2: REST API GATEWAY (Flask Server on Render / Vercel)",
            "#0284c7",
            "<b>Endpoint:</b> <code>POST /ocr</code> &nbsp;|&nbsp; <b>Port:</b> 5000 / 443 HTTPS<br/>"
            "<b>Accepted Formats:</b> Multipart Form (<code>file</code> key) OR JSON (<code>image</code> Base64).<br/>"
            "<b>Sub-millisecond Routing:</b> Fast pre-check, payload decoding, and auto-dispatch to scanner."
        ),
        (
            "STEP 3: LAB-SPACE CLAHE CONTRAST & DUAL-VIEW PIPELINE",
            "#d97706",
            "<b>CLAHE Contrast Enhancement:</b> Converts BGR to LAB color space; applies Contrast Limited Adaptive "
            "Histogram Equalization (clip=3.0) on L-channel. Eliminates floor/background color confusion inside cutouts.<br/>"
            "<b>Dual-View Generation:</b> Assembles <b>View 1 (Original)</b> and <b>View 2 (Horizontally Flipped)</b> into a single visual prompt."
        ),
        (
            "STEP 4: MULTI-KEY ROUND-ROBIN & DYNAMIC FAILOVER AI ENGINE",
            "#7c3aed",
            "<b>Primary Vision Model:</b> <code>gemini-2.5-flash</code> (ultra-fast 1.2s inference).<br/>"
            "<b>Fallback Chain:</b> <code>gemini-flash-latest</code> &nbsp;➔&nbsp; <code>gemini-3-flash-preview</code> &nbsp;➔&nbsp; <code>gemini-3.8-flash</code>.<br/>"
            "<b>Active Key Pool:</b> 6+ keys in circular round-robin. On HTTP 429 (rate limit) or 401, key gets 60s cooldown; request instantly auto-fails over to next active key."
        ),
        (
            "STEP 5: IMPLEMENT CATALOG NORMALIZER & BRIDGE HEALING",
            "#059669",
            "<b>Order Reversal:</b> If code is read backwards (e.g. <code>UP/.../GJ</code>), tokens are automatically reversed.<br/>"
            "<b>State Normalization:</b> Corrects bridge-cut 'G' (<code>CJ</code>, <code>6J</code>, <code>0J</code> ➔ <code>GJ</code>) or prepends missing state.<br/>"
            "<b>Model Catalog:</b> Maps aliases (<code>44G</code>➔<code>446</code>, <code>367A</code>➔<code>3674</code>, <code>44S</code>➔<code>445</code>).<br/>"
            "<b>Serial Repair:</b> Eliminates ghost digits from stencil bridge in '8', maps letters (O➔0, I➔1), ensures 5-digit format."
        ),
        (
            "STEP 6: MINIMAL HIGH-SPEED JSON RESPONSE",
            "#0f172a",
            "<b>Response Payload:</b> <code>{\"status\": \"SUCCESS\", \"PARTno\": \"GJ/142/446/2026/11881/UP\"}</code><br/>"
            "<b>Developer Experience:</b> Clean, instant, and ready to consume by mobile app or ERP backend."
        )
    ]

    for idx, (title, color_hex, desc) in enumerate(flow_steps):
        # Card table
        card_content = [
            [Paragraph(f"<b>{title}</b>", ParagraphStyle("H", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor(color_hex)))],
            [Paragraph(desc, step_desc)]
        ]
        card_table = Table(card_content, colWidths=[528])
        card_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(color_hex)),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(card_table)

        # Connector Arrow (except after last step)
        if idx < len(flow_steps) - 1:
            arrow_d = Drawing(532, 14)
            arrow_d.add(Line(266, 14, 266, 2, strokeColor=colors.HexColor("#94a3b8"), strokeWidth=1.5))
            arrow_d.add(Polygon([262, 4, 270, 4, 266, 0], fillColor=colors.HexColor("#64748b"), strokeColor=None))
            story.append(arrow_d)

    story.append(Spacer(1, 10))

    # Page Break for Deep Dive & Physical Diagram
    story.append(PageBreak())

    story.append(Paragraph("Physical Stencil Anatomy & Stencil Bridge Mechanics", section_heading))
    story.append(Paragraph(
        "Why standard OCR fails and how our specialized engine restores laser-cut implement codes:",
        subtitle_style
    ))
    story.append(Spacer(1, 6))

    anatomy_data = [
        [
            Paragraph("<b>Physical Characteristic</b>", step_title),
            Paragraph("<b>Standard OCR Failure Mode</b>", step_title),
            Paragraph("<b>Our Engine's Solution</b>", step_title)
        ],
        [
            Paragraph("<b>Laser Through-Cut Slits</b><br/>Holes cut completely through sheet metal", step_desc),
            Paragraph("Background floor, wooden pallets, rust, or shadows show through slits; model confuses colors.", step_desc),
            Paragraph("<b>LAB CLAHE Preprocessing</b> isolates local luminance, removing background color hue and glare.", step_desc)
        ],
        [
            Paragraph("<b>Stencil Font Bridge Tabs</b><br/>Metal tie-bars across 0, 4, 6, 8, 9, A, B, G", step_desc),
            Paragraph("Broken character contours cause misreads: '4' ➔ 'A', '6' ➔ 'G', '8' ➔ '38', 'G' ➔ 'C'.", step_desc),
            Paragraph("<b>Dual-View + Bridge Normalizer</b> heals broken contours and maps catalog aliases automatically.", step_desc)
        ],
        [
            Paragraph("<b>Mirrored Reverse Shots</b><br/>Metal photographed from back side", step_desc),
            Paragraph("Inverted text (e.g. 11541 ➔ 12476) results in complete hallucination.", step_desc),
            Paragraph("<b>Dual-View Generation</b> presents original + horizontally flipped views; AI auto-selects forward read.", step_desc)
        ],
        [
            Paragraph("<b>Reversed Token Order</b><br/>Upside-down photo reads right-to-left", step_desc),
            Paragraph("Produces 'UP/.../GJ'; regex expecting leading state fails to match.", step_desc),
            Paragraph("<b>Auto-Reversal</b> detects trailing state ('GJ' at end) and flips token array into canonical order.", step_desc)
        ]
    ]

    anatomy_table = Table(anatomy_data, colWidths=[160, 180, 188])
    anatomy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(anatomy_table)

    story.append(Spacer(1, 14))

    # Production API Quick Reference
    story.append(Paragraph("Live Production Endpoints & Integration Reference", section_heading))
    
    spec_data = [
        [Paragraph("<b>Production Render URL</b>", step_title), Paragraph("<code>https://s-ocr-3zbt.onrender.com/ocr</code>", code_style)],
        [Paragraph("<b>Vercel Edge URL</b>", step_title), Paragraph("<code>https://s-ocr.vercel.app/api/ocr</code>", code_style)],
        [Paragraph("<b>HTTP Methods</b>", step_title), Paragraph("<b>POST</b> (Process Image) &nbsp;|&nbsp; <b>GET</b> (Interactive Doc/Test)", step_desc)],
        [Paragraph("<b>Supported Payloads</b>", step_title), Paragraph("Multipart form-data (<code>file</code> or <code>image</code>) &nbsp;|&nbsp; JSON (<code>image</code> base64)", step_desc)],
        [Paragraph("<b>Response Schema</b>", step_title), Paragraph("<code>{\"status\": \"SUCCESS\", \"PARTno\": \"GJ/142/446/2026/11881/UP\"}</code>", code_style)],
        [Paragraph("<b>Error Schema</b>", step_title), Paragraph("<code>{\"status\": \"FAILED\", \"PARTno\": null, \"error\": \"...\"}</code>", code_style)],
        [Paragraph("<b>Failover Pool</b>", step_title), Paragraph("6+ Keys Round-Robin with 60s cooldown auto-switch on HTTP 429", step_desc)],
    ]
    spec_table = Table(spec_data, colWidths=[150, 378])
    spec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(spec_table)

    doc.build(story, canvasmaker=FlowchartCanvas)
    print(f"Successfully generated {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
