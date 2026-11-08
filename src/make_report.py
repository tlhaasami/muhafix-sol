"""
Muhafiz.sol - final report (.docx) and slides (.pptx) generated from the result
tables, so the numbers in the report, the slides and the app are identical
(blueprint 19, gate G4).

    python -m src.make_report

Outputs: reports/Muhafiz_sol_Final_Report.docx, reports/Muhafiz_sol_Slides.pptx,
docs/model_card.md, docs/data_card.md, docs/dedup_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from src.config import ROOT, path
from src.report_utils import MODEL_NAMES, Facts, fmt, pct

FIG = path("figures_dir")
ACCENT = RGBColor(0x1F, 0x4E, 0x8C)
GREY = RGBColor(0x52, 0x51, 0x4E)


# --------------------------------------------------------------------------- #
# docx helpers
# --------------------------------------------------------------------------- #
class Doc:
    def __init__(self):
        self.d = Document()
        sec = self.d.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.2))
        sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
        st = self.d.styles
        st["Normal"].font.name = "Calibri"
        st["Normal"].font.size = Pt(11)
        st["Normal"].paragraph_format.space_after = Pt(6)
        st["Normal"].paragraph_format.line_spacing = 1.12
        for lvl, size in ((1, 17), (2, 13.5), (3, 12)):
            h = st[f"Heading {lvl}"]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = ACCENT
            h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
            h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True
        self.fig_no = 0
        self.tab_no = 0
        self._footer()

    def _footer(self):
        p = self.d.sections[0].footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run("Muhafiz.sol - AI2002 Group 39 - page ")
        run.font.size = Pt(8)
        run.font.color.rgb = GREY
        self._field(p, "PAGE")

    @staticmethod
    def _field(paragraph, instr: str):
        run = paragraph.add_run()
        for tag, text in (("begin", None), (None, instr), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)
        run.font.size = Pt(8)

    def h(self, text: str, level: int = 1):
        self.d.add_heading(text, level)

    def p(self, text: str = "", bold_lead: Optional[str] = None, italic: bool = False, size: Optional[float] = None,
          align=None, color=None):
        para = self.d.add_paragraph()
        if bold_lead:
            r = para.add_run(bold_lead)
            r.bold = True
        r = para.add_run(text)
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
        if align:
            para.alignment = align
        return para

    def bullets(self, items: Iterable[str], style: str = "List Bullet"):
        for it in items:
            para = self.d.add_paragraph(style=style)
            if isinstance(it, tuple):
                r = para.add_run(it[0])
                r.bold = True
                para.add_run(it[1])
            else:
                para.add_run(it)
            para.paragraph_format.space_after = Pt(2)

    def numbered(self, items: Iterable[str]):
        """Numbered list that restarts at 1 (Word would otherwise continue the previous list's numbering)."""
        items = list(items)
        self.bullets(items, "List Number")
        num_pr = self.d.styles["List Number"].element.pPr.numPr if self.d.styles["List Number"].element.pPr is not None else None
        if num_pr is None or num_pr.numId is None:
            return
        numbering = self.d.part.numbering_part.element
        abstract_id = numbering.num_having_numId(num_pr.numId.val).abstractNumId.val
        new_num = numbering.add_num(abstract_id)
        new_num.add_lvlOverride(ilvl=0).add_startOverride(1)
        for para in self.d.paragraphs[-len(items):]:
            p_num = para._p.get_or_add_pPr().get_or_add_numPr()
            p_num.get_or_add_ilvl().val = 0
            p_num.get_or_add_numId().val = new_num.numId

    def table(self, df: pd.DataFrame, caption: str, nd: int = 3, widths: Optional[Sequence[float]] = None,
              font: float = 9):
        self.tab_no += 1
        cap = self.d.add_paragraph()
        r = cap.add_run(f"Table {self.tab_no}: {caption}")
        r.bold = True
        r.font.size = Pt(9.5)
        cap.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(df.columns))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, c in enumerate(df.columns):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(c))
            run.bold = True
            run.font.size = Pt(font)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, c in enumerate(df.columns):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(fmt(row[c], nd))
                run.font.size = Pt(font)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def figure(self, file: Path, caption: str, width_cm: float = 16.0):
        if not Path(file).exists():
            self.p(f"[missing figure: {Path(file).name}]", italic=True)
            return
        self.fig_no += 1
        self.d.add_picture(str(file), width=Cm(width_cm))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = self.p(f"Figure {self.fig_no}: {caption}", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
                     color=GREY)
        cap.paragraph_format.space_after = Pt(10)

    def page_break(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def toc(self):
        self._toc_anchor = self.d.add_paragraph()
        self.p("(Page numbers appear when Word updates the field: right-click the list and choose Update Field.)",
               italic=True, size=8, color=GREY)

    def _fill_toc(self):
        """Write a TOC field whose cached result already lists the chapter headings, so the contents page is
        filled even before Word recomputes it (python-docx cannot compute page numbers)."""
        anchor = getattr(self, "_toc_anchor", None)
        if anchor is None:
            return
        heads, after = [], False
        for para in self.d.paragraphs:
            if para._p is anchor._p:
                after = True
            elif after and para.style.name in ("Heading 1", "Heading 2") and para.text.strip():
                heads.append((1 if para.style.name == "Heading 1" else 2, para.text.strip()))
        if not heads:
            return

        def fld(run, kind=None, instr=None):
            if instr is not None:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = instr
            else:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), kind)
            run._r.append(el)

        paras = []
        for level, text in heads:
            para = anchor.insert_paragraph_before()
            para.paragraph_format.left_indent = Cm(0.6 * (level - 1))
            para.paragraph_format.space_after = Pt(1)
            paras.append((para, level, text))
        first = paras[0][0]
        fld(first.add_run(), "begin")
        fld(first.add_run(), instr='TOC \\o "1-2" \\h \\z \\u')
        fld(first.add_run(), "separate")
        for para, level, text in paras:
            run = para.add_run(text)
            run.bold = level == 1
            run.font.size = Pt(10 if level == 1 else 9.5)
        fld(paras[-1][0].add_run(), "end")
        anchor._p.getparent().remove(anchor._p)
        self._toc_anchor = None

    def save(self, p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fill_toc()
        self.d.save(str(p))


# --------------------------------------------------------------------------- #
def appendix(doc: Doc, f: Facts) -> None:
    doc.h("Appendix A - Compliance checklist (blueprint Part B)")
    rows = [
        ["Exactly one path; no A/B mixture", "Path A only; no hardware", "met"],
        ["Group of 2-3 members", "3 members (Group 39)", "met"],
        ["Core functionality depends on ML", "every output comes from trained models; no rules at prediction time", "met"],
        ["Not a CRUD / rule-based system", "Slither only produced training labels; retraining changes outputs", "met"],
        ["Algorithm justified for the use case", "chapters 1.3 and 4", "met"],
        ["Dataset documented (source, features, target, preprocessing, limitations)", "chapter 3, docs/data_card.md", "met"],
        ["Pipeline evaluated with appropriate metrics and interpreted", "chapter 5 (E1-E4), chapter 7 (E3b, E5)", "met"],
        ["Streamlit deployment with the complete flow", "app/app.py: input -> model -> explained result", "met"],
        ["Tested with representative inputs", "test split, SmartBugs, ten fixed inputs, fuzzing, pytest", "met"],
        ["Report chapters required by the statement", "chapters 1-9 of this report", "met"],
        ["Presentation slides", "reports/Muhafiz_sol_Slides.pptx (20 slides)", "met"],
        ["Independent external test", "SmartBugs never used for training or tuning; 42 of 143 contracts overlap the "
                                       "training pool, so E3/E3b also report the 51 + 18 clean contracts", "partly"],
        ["Integrity: commit history per member, reviewed pull requests", "all commits were made from one account "
                                       "(tlhaasami) in one local repository; no pull requests; contributions per member "
                                       "are listed in chapter 9", "partly"],
        ["One-command rebuild on another laptop (A6)", "python -m src.build_data run on the development laptop only",
         "pending (team)"],
        ["Live demonstration", "rehearsal on the demo laptop", "pending (team)"],
        ["Every member can explain every part", "mock viva", "pending (team)"],
        ["One-command run on all three laptops", "verified on the development laptop", "pending (two laptops)"],
        ["Single zip submitted on GCR", "-", "pending (team, 6 Dec)"],
    ]
    doc.table(pd.DataFrame(rows, columns=["Requirement", "Evidence", "Status"]), "Compliance with the project statement",
              widths=[6.5, 7.5, 2.8], font=8)
    doc.h("Appendix B - User guide")
    doc.numbered([
        "Install once: python -m venv .venv, activate it, pip install -r requirements.txt.",
        "Start: streamlit run app/app.py - the browser opens at http://localhost:8501.",
        "Paste a contract into the text box, upload a .sol file, or pick a sample in the sidebar; press Scan.",
        "Read the risk badge and the bars: a bar that crosses its dark tick is flagged.",
        "Choose a weakness type under 'Where in the code?' to see its plain-language meaning and the highlighted lines.",
        "Optionally move the sensitivity slider (left = more alerts), switch to logistic regression, or download the report.",
        "Use 'Scan many files' to rank several contracts by risk. Remember: a low score is not a safety certificate.",
    ])
    doc.p("The model card (docs/model_card.md) and data card (docs/data_card.md) are part of the submission and are "
          "regenerated from the same tables as this report; they are reproduced in Appendices C and D.")
    for letter, title, name in (("C", "Model card", "model_card.md"), ("D", "Data card", "data_card.md")):
        card = path("docs_dir") / name
        if card.exists():
            doc.page_break()
            doc.h(f"Appendix {letter} - {title}")
            _markdown(doc, card.read_text(encoding="utf-8"), title.lower())


def _markdown(doc: "Doc", text: str, source: str = "card") -> None:
    """Very small Markdown renderer for the two cards: headings, tables, bullets and paragraphs."""
    lines, i = text.splitlines(), 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("#"):
            level = len(ln) - len(ln.lstrip("#"))
            if level > 1:  # the card's own title is replaced by the appendix heading
                doc.h(ln.lstrip("#").strip(), 3)
            i += 1
        elif ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append([c.replace("**", "") for c in cells])
                i += 1
            if len(rows) > 1:
                doc.table(pd.DataFrame(rows[1:], columns=rows[0]), f"From the {source}", font=8)
        elif ln.lstrip().startswith(("* ", "- ")):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith(("* ", "- ")):
                item = lines[i].lstrip()[2:].strip()
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and not lines[i].lstrip().startswith(("* ", "- ")):
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item.replace("**", "").replace("`", ""))
            doc.bullets(items)
        else:
            para = [ln.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("#", "|", "* ", "- ")):
                para.append(lines[i].strip())
                i += 1
            doc.p(" ".join(para).replace("**", "").replace("`", ""))


def build_report() -> Path:
    from src.report_part1 import chapter1, chapter2, chapter3, front_matter
    from src.report_part2 import chapter4, chapter5, chapter7, chapter9
    from src.report_part3 import chapter6, chapter8, references

    f = Facts()
    doc = Doc()
    front_matter(doc, f)
    for ch in (chapter1, chapter2, chapter3, chapter4, chapter5, chapter6, chapter7, chapter8, chapter9):
        ch(doc, f)
        doc.page_break()
    references(doc)
    doc.page_break()
    appendix(doc, f)
    out = path("reports_dir") / "Muhafiz_sol_Final_Report.docx"
    doc.save(out)
    return out


if __name__ == "__main__":
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "src.make_docs", "--model"], check=True)
    print("report:", build_report())
    from src.make_slides import build_slides

    print("slides:", build_slides())
