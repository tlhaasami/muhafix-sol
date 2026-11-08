"""
Muhafiz.sol - presentation slides (blueprint 19.2: about 20 slides) generated
with python-pptx from the same result tables as the report.

    python -m src.make_slides   ->  reports/Muhafiz_sol_Slides.pptx
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

from src.config import path
from src.report_utils import MODEL_NAMES, Facts, pct

NAVY = RGBColor(0x14, 0x23, 0x4B)
TEAL = RGBColor(0x0E, 0x8A, 0x8A)
AMBER = RGBColor(0xF2, 0xA5, 0x41)
RED = RGBColor(0xC6, 0x3B, 0x34)
INK = RGBColor(0x1F, 0x1F, 0x1F)
MUTED = RGBColor(0x5F, 0x63, 0x68)
LIGHT = RGBColor(0xF3, 0xF6, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"
FIG = path("figures_dir")
W, H = Inches(13.333), Inches(7.5)


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    # ----------------------------------------------------------- primitives --
    def slide(self, dark: bool = False):
        s = self.prs.slides.add_slide(self.blank)
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = NAVY if dark else WHITE
        self.n += 1
        if not dark:
            self.text(s, f"{self.n}", Inches(12.55), Inches(7.02), Inches(0.5), Inches(0.3), 10, MUTED,
                      align=PP_ALIGN.RIGHT)
            self.text(s, "Muhafiz.sol - AI2002 Group 39", Inches(0.6), Inches(7.02), Inches(5), Inches(0.3), 10, MUTED)
        return s

    def text(self, s, txt, x, y, w, h, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, italic=False):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Emu(0)
        lines = txt if isinstance(txt, list) else [txt]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = line
            r.font.size, r.font.bold, r.font.italic = Pt(size), bold, italic
            r.font.color.rgb, r.font.name = color, font
        return tb

    def title(self, s, txt, sub: Optional[str] = None):
        self.text(s, txt, Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.8), 34, NAVY, True, HEAD)
        if sub:
            self.text(s, sub, Inches(0.6), Inches(1.12), Inches(12.1), Inches(0.5), 16, MUTED, italic=True)

    def bullets(self, s, items: Sequence, x, y, w, h, size=16, color=INK, gap=8):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Emu(0)
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(gap)
            lead, rest = (it if isinstance(it, tuple) else ("", it))
            r0 = p.add_run()
            r0.text = "▸  "
            r0.font.color.rgb, r0.font.size = TEAL, Pt(size)
            if lead:
                r = p.add_run()
                r.text = lead
                r.font.bold, r.font.size, r.font.color.rgb, r.font.name = True, Pt(size), color, BODY
            r = p.add_run()
            r.text = rest
            r.font.size, r.font.color.rgb, r.font.name = Pt(size), color, BODY
        return tb

    def card(self, s, x, y, w, h, head, body, num: Optional[str] = None, fill=LIGHT, head_color=NAVY):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        box.adjustments[0] = 0.08
        box.fill.solid()
        box.fill.fore_color.rgb = fill
        box.line.fill.background()
        box.shadow.inherit = False
        off = Inches(0.25)
        if num:
            c = s.shapes.add_shape(MSO_SHAPE.OVAL, x + off, y + off, Inches(0.55), Inches(0.55))
            c.fill.solid()
            c.fill.fore_color.rgb = TEAL
            c.line.fill.background()
            tf = c.text_frame
            tf.paragraphs[0].alignment = PP_ALIGN.CENTER
            r = tf.paragraphs[0].add_run()
            r.text = num
            r.font.size, r.font.bold, r.font.color.rgb = Pt(16), True, WHITE
            hx = x + off + Inches(0.7)
        else:
            hx = x + off
        self.text(s, head, hx, y + off, w - (hx - x) - off, Inches(0.55), 18, head_color, True, HEAD,
                  anchor=MSO_ANCHOR.MIDDLE)
        self.text(s, body, x + off, y + off + Inches(0.7), w - 2 * off, h - Inches(1.05), 14, INK)

    def stat(self, s, x, y, w, value, label, color=TEAL, size=44):
        self.text(s, value, x, y, w, Inches(0.95), size, color, True, HEAD, anchor=MSO_ANCHOR.BOTTOM)
        self.text(s, label, x, y + Inches(0.95), w, Inches(0.8), 13, MUTED)

    def image(self, s, file: Path, x, y, w=None, h=None):
        if Path(file).exists():
            return s.shapes.add_picture(str(file), x, y, w, h)
        self.text(s, f"[missing {Path(file).name}]", x, y, Inches(4), Inches(0.4), 12, RED)

    def table(self, s, df: pd.DataFrame, x, y, w, row_h=0.36, size=12, col_w: Optional[List[float]] = None):
        rows, cols = len(df) + 1, len(df.columns)
        t = s.shapes.add_table(rows, cols, x, y, w, Inches(row_h * rows)).table
        for j, c in enumerate(df.columns):
            cell = t.cell(0, j)
            cell.text = str(c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY
            para = cell.text_frame.paragraphs[0]
            para.runs[0].font.size, para.runs[0].font.bold = Pt(size), True
            para.runs[0].font.color.rgb = WHITE
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            for j, c in enumerate(df.columns):
                v = r[c]
                txt = v if isinstance(v, str) else (f"{v:.3f}" if isinstance(v, float) else str(v))
                cell = t.cell(i, j)
                cell.text = txt
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
                run = cell.text_frame.paragraphs[0].runs[0]
                run.font.size, run.font.color.rgb = Pt(size), INK
        if col_w:
            for j, cw in enumerate(col_w):
                t.columns[j].width = Inches(cw)
        return t

    def bar_chart(self, s, categories, series: dict, x, y, w, h, title: str, colors, maxv=1.0, fmt='0.00'):
        cd = CategoryChartData()
        cd.categories = list(categories)
        for name, vals in series.items():
            cd.add_series(name, [float(v) for v in vals])
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, w, h, cd)
        ch = gf.chart
        ch.has_title = True
        ch.chart_title.text_frame.text = title
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
        ch.has_legend = len(series) > 1
        if ch.has_legend:
            ch.legend.position = XL_LEGEND_POSITION.TOP
            ch.legend.include_in_layout = False
            ch.legend.font.size = Pt(11)
        va = ch.value_axis
        va.maximum_scale, va.minimum_scale = maxv, 0
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xE3, 0xDF)
        va.tick_labels.font.size = Pt(10)
        va.tick_labels.font.color.rgb = MUTED
        va.format.line.fill.background()
        ca = ch.category_axis
        ca.tick_labels.font.size = Pt(10)
        ca.tick_labels.font.color.rgb = MUTED
        plot = ch.plots[0]
        plot.gap_width = 60
        plot.overlap = -5
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format, dl.number_format_is_linked = fmt, False
        dl.font.size = Pt(9)
        dl.font.color.rgb = INK
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        for ser, col in zip(plot.series, colors):
            ser.format.fill.solid()
            ser.format.fill.fore_color.rgb = col
        return ch

    def notes(self, s, txt: str):
        s.notes_slide.notes_text_frame.text = txt

    def save(self, p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(p))


def build_slides() -> Path:
    from src.slides_content import build

    deck = Deck()
    build(deck, Facts())
    out = path("reports_dir") / "Muhafiz_sol_Slides.pptx"
    deck.save(out)
    return out


if __name__ == "__main__":
    print(build_slides())
