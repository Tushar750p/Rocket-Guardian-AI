"""Professional ReportLab theme for Rocket Guardian AI customer reports."""

import reportlab.platypus as _platypus
import reportlab.lib.styles as _styles_module
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm

# Preserve the true ReportLab objects exactly once.  Streamlit can reload
# modules between reruns; without these sentinels a reload would capture our
# own patched Paragraph class and recurse forever.
if not hasattr(_platypus, "_rg_original_paragraph"):
    _platypus._rg_original_paragraph = _platypus.Paragraph
if not hasattr(_platypus, "_rg_original_simple_doc_template"):
    _platypus._rg_original_simple_doc_template = _platypus.SimpleDocTemplate
if not hasattr(_styles_module, "_rg_original_get_sample_styles"):
    _styles_module._rg_original_get_sample_styles = _styles_module.getSampleStyleSheet

_OriginalParagraph = _platypus._rg_original_paragraph
_OriginalSimpleDocTemplate = _platypus._rg_original_simple_doc_template
_original_get_styles = _styles_module._rg_original_get_sample_styles


_PALETTE = {
    "navy": colors.HexColor("#0F172A"),
    "slate": colors.HexColor("#475569"),
    "line": colors.HexColor("#CBD5E1"),
    "soft": colors.HexColor("#F8FAFC"),
    "accent": colors.HexColor("#2563EB"),
}


class _ProfessionalDocTemplate(_OriginalSimpleDocTemplate):
    """SimpleDocTemplate with a branded header/footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = "Rocket Guardian AI - Telemetry Analysis Report"
        self.author = "Rocket Guardian AI"
        self.subject = "Telemetry risk analysis"

    def build(self, flowables, *args, **kwargs):
        def draw_page(canvas, document):
            canvas.saveState()
            width, height = document.pagesize

            canvas.setStrokeColor(_PALETTE["line"])
            canvas.setLineWidth(0.6)
            canvas.line(
                document.leftMargin,
                height - 18 * mm,
                width - document.rightMargin,
                height - 18 * mm,
            )
            canvas.setFillColor(_PALETTE["navy"])
            canvas.setFont("Helvetica-Bold", 9)
            canvas.drawString(
                document.leftMargin,
                height - 13 * mm,
                "ROCKET GUARDIAN AI",
            )
            canvas.setFillColor(_PALETTE["slate"])
            canvas.setFont("Helvetica", 8)
            canvas.drawRightString(
                width - document.rightMargin,
                height - 13 * mm,
                "TELEMETRY ANALYSIS REPORT",
            )

            canvas.setStrokeColor(_PALETTE["line"])
            canvas.line(
                document.leftMargin,
                17 * mm,
                width - document.rightMargin,
                17 * mm,
            )
            canvas.setFillColor(_PALETTE["slate"])
            canvas.setFont("Helvetica", 7.5)
            canvas.drawString(
                document.leftMargin,
                11 * mm,
                "Research Prototype - Not for flight-critical use",
            )
            canvas.drawRightString(
                width - document.rightMargin,
                11 * mm,
                f"Page {canvas.getPageNumber()}",
            )
            canvas.restoreState()

        kwargs.pop("onFirstPage", None)
        kwargs.pop("onLaterPages", None)
        return super().build(
            flowables,
            onFirstPage=draw_page,
            onLaterPages=draw_page,
            *args,
            **kwargs,
        )


class _ProfessionalStyles:
    """Proxy style sheet retaining standard names used by the dashboard."""

    def __init__(self):
        styles = _original_get_styles()
        self.styles = styles

        self.title = styles["Title"]
        self.title.fontName = "Helvetica-Bold"
        self.title.fontSize = 22
        self.title.leading = 26
        self.title.textColor = _PALETTE["navy"]
        self.title.alignment = TA_CENTER
        self.title.spaceAfter = 10

        self.heading2 = styles["Heading2"]
        self.heading2.fontName = "Helvetica-Bold"
        self.heading2.fontSize = 13
        self.heading2.leading = 16
        self.heading2.textColor = _PALETTE["navy"]
        self.heading2.spaceBefore = 8
        self.heading2.spaceAfter = 8

        body = styles["BodyText"]
        body.fontName = "Helvetica"
        body.fontSize = 9
        body.leading = 13
        body.textColor = colors.HexColor("#1E293B")
        body.spaceAfter = 4

        self.section = ParagraphStyle(
            "RGSection",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=_PALETTE["accent"],
            spaceBefore=10,
            spaceAfter=5,
        )

        self.metric = ParagraphStyle(
            "RGMetric",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=13,
            textColor=_PALETTE["navy"],
            backColor=_PALETTE["soft"],
            borderColor=_PALETTE["line"],
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=2,
            spaceAfter=4,
        )

        self.small = ParagraphStyle(
            "RGSmall",
            parent=body,
            fontSize=8,
            leading=11,
            textColor=_PALETTE["slate"],
        )

    def __getitem__(self, key):
        if key == "Title":
            return self.title
        if key == "Heading2":
            return self.heading2
        if key == "BodyText":
            return self.styles["BodyText"]
        return self.styles[key]


def _professional_get_stylesheet():
    return _ProfessionalStyles()


def _professional_paragraph(text, style=None, *args, **kwargs):
    raw = str(text).strip()
    styles = _ProfessionalStyles()

    if raw in {
        "MISSION SUMMARY",
        "RISK ASSESSMENT",
        "SENSOR RISK",
        "PEAK RISK EVENT",
        "EXPLANATION",
    }:
        style = styles.section
    elif raw.startswith("Rocket Guardian AI - Research Prototype"):
        style = styles.small
    elif raw.startswith("Ground-truth anomaly labels") or raw.startswith("AI detections are based"):
        style = styles.small

    return _OriginalParagraph(text, style, *args, **kwargs)


# Patch only the public module attributes. The original objects are retained
# above so subsequent Streamlit reloads remain safe and non-recursive.
_platypus.SimpleDocTemplate = _ProfessionalDocTemplate
_platypus.Paragraph = _professional_paragraph
_styles_module.getSampleStyleSheet = _professional_get_stylesheet
