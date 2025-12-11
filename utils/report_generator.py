# report_generator.py
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import datetime
import os
from textwrap import wrap

def generate_report(task_type, model_name, scores, image_paths, steps_text, out_path="reports/ML_Report.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "AutoML Model Report")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Model: {model_name}")
    y -= 18
    c.drawString(50, y, f"Task: {task_type}")
    y -= 18
    c.drawString(50, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 24

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Summary Steps")
    y -= 18
    c.setFont("Helvetica", 10)
    wrapped = wrap(steps_text, 100)
    for line in wrapped:
        if y < 80:
            c.showPage(); y = height - 50
        c.drawString(50, y, line)
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Evaluation Metrics")
    y -= 18
    c.setFont("Helvetica", 11)
    if not scores:
        c.drawString(60, y, "No scores available.")
        y -= 14
    else:
        for k, v in scores.items():
            if y < 80:
                c.showPage(); y = height - 50
            c.drawString(60, y, f"- {k}: {v}")
            y -= 14

    # images each on new page
    for label, path in image_paths.items():
        if not path or not os.path.exists(path):
            continue
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 50, label)
        try:
            img = ImageReader(path)
            img_w = width - 100
            img_h = height - 150
            c.drawImage(img, 50, 80, width=img_w, height=img_h, preserveAspectRatio=True, anchor='c')
        except Exception:
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 80, f"[Could not render image: {path}]")

    c.save()
    return out_path
