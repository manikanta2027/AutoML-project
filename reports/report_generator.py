# report_generator.py – FULL CODE EXPORT VERSION

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from textwrap import wrap
import datetime, os


# ==========================================
# MAIN REPORT BUILDER
# ==========================================
def generate_report(task_type, model_name, scores, image_paths, full_code,
                    out_path="reports/ML_Report.pdf"):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    

    # ================= HEADER PAGE =================
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height-70, "📄 AutoML Research Report")

    c.setFont("Helvetica", 13)
    c.drawString(50, height-110, f"• Model Used      : {model_name}")
    c.drawString(50, height-130, f"• Task Type       : {task_type}")
    c.drawString(50, height-150, f"• Generated On    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, height-190, "🔹 Performance Metrics")
    c.setFont("Helvetica", 11)

    y = height-220
    if scores:
        for m,v in scores.items():
            c.drawString(60, y, f"- {m}: {v:.4f}")
            y -= 16
    else:
        c.drawString(60, y, "No evaluation metrics available."); y -= 16

    c.showPage()


    # ================= IMAGES =================
    for title, path in image_paths.items():
        if os.path.exists(path):

            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height-60, f"📌 {title}")

            img = ImageReader(path)
            c.drawImage(img, 1*inch, 1.2*inch, width=5.5*inch, preserveAspectRatio=True)
            c.showPage()


    # ================= FULL CODE SECTION =================
    write_full_code(c, model_name, full_code)

    c.save()
    return out_path



# ==========================================
# FUNCTION THAT PRINTS CLEAN MULTI-PAGE CODE
# ==========================================
def write_full_code(c, model_name, code):

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, f"💻 Full Source Code — {model_name}")

    c.setFont("Courier", 9)  # Mono font for real code formatting
    y = 770

    for line in wrap(code, 110):   # 110 characters per line output
        c.drawString(50, y, line)
        y -= 12

        if y < 60:   # Next page
            c.showPage()
            c.setFont("Courier", 9)
            y = 780
