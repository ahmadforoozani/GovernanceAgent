from datetime import datetime, timezone
from pathlib import Path
import markdown

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor


def _header_footer(canvas, doc):
    canvas.saveState()

    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(40, 800, "Governance Audit System")

    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(550, 800, datetime.now(
        timezone.utc).strftime("%Y-%m-%d"))

    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(300, 20, f"Page {doc.page}")

    canvas.restoreState()


def build_pdf(md_path: str, output_path: str):

    md_text = Path(md_path).read_text(encoding="utf-8")

    html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=22,
        textColor=HexColor("#1F4E79"),
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=HexColor("#2F5597"),
        spaceAfter=12
    )

    heading2_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading3"],
        fontSize=11,
        textColor=HexColor("#366B8A"),
        spaceAfter=10
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        spaceAfter=6
    )

    story = []

    for line in html.split("\n"):

        line = line.strip()

        if not line:
            story.append(Spacer(1, 8))
            continue

        if line.startswith("<h1>"):
            text = line.replace("<h1>", "").replace("</h1>", "")
            story.append(Paragraph(text, title_style))

        elif line.startswith("<h2>"):
            text = line.replace("<h2>", "").replace("</h2>", "")
            story.append(Paragraph(text, heading_style))

        elif line.startswith("<h3>"):
            text = line.replace("<h3>", "").replace("</h3>", "")
            story.append(Paragraph(text, heading2_style))

        else:
            story.append(Paragraph(line, body_style))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=80,
        bottomMargin=40
    )

    doc.build(
        story,
        onFirstPage=_header_footer,
        onLaterPages=_header_footer
    )

    return output_path
