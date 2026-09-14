"""Скан-PDF описывается как скан, а не как обычный документ.

Извлечение текста из скана возвращает пустоту, и агент читает её как «документ пустой»:
присланный файл молча выпадает из анализа. Так и произошло с медицинским сканом среди
21 PDF — он был отмечен «почти пустой» и не прочитан. Заметка должна называть вещи
своими именами и указывать путь через рендер страниц.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from gateway.run import _build_document_context_note, _document_is_scanned_pdf

pytestmark = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")


def _pdf(path, *, text: str | None) -> str:
    """Minimal one-page PDF, with a text layer or with none.

    A text page carries a realistic amount of prose: the check measures text density, so a
    page holding one short phrase is legitimately indistinguishable from page furniture.
    """
    if text:
        body = chr(10).join(
            f"BT /F1 11 Tf 72 {720 - 14 * i} Td ({text} - line {i:02d} of the report) Tj ET"
            for i in range(40)
        )
    else:
        body = "72 700 m 200 700 l S"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(body)} >>\nstream\n{body}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = "%PDF-1.4\n"
    for i, o in enumerate(objs, start=1):
        out += f"{i} 0 obj\n{o}\nendobj\n"
    out += f"trailer\n<< /Root 1 0 R /Size {len(objs) + 1} >>\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))
    return str(path)


def test_pdf_without_text_layer_is_called_scanned(tmp_path):
    path = _pdf(tmp_path / "scan.pdf", text=None)
    assert _document_is_scanned_pdf(path, "application/pdf") is True
    note = _build_document_context_note("scan.pdf", path, "application/pdf")
    assert "SCANNED" in note and "NO text layer" in note
    assert "vision_analyze" in note
    assert "does not mean the document is empty" in note


def test_pdf_with_text_layer_gets_the_ordinary_note(tmp_path):
    path = _pdf(tmp_path / "report.pdf", text="Hemoglobin 140 g per L within range")
    assert _document_is_scanned_pdf(path, "application/pdf") is False
    note = _build_document_context_note("report.pdf", path, "application/pdf")
    assert "SCANNED" not in note


def test_non_pdf_is_never_probed(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_bytes(b"PK\x03\x04 not a pdf")
    assert _document_is_scanned_pdf(str(path), "application/vnd.openxmlformats-officedocument") is False


def test_missing_file_does_not_raise(tmp_path):
    assert _document_is_scanned_pdf(str(tmp_path / "gone.pdf"), "application/pdf") is False


def test_note_still_built_when_probe_is_impossible(tmp_path, monkeypatch):
    """No pdftotext on the host: fall back to the ordinary note rather than guessing."""
    monkeypatch.setattr("shutil.which", lambda _name: None)
    path = _pdf(tmp_path / "scan.pdf", text=None)
    assert _document_is_scanned_pdf(path, "application/pdf") is False
    assert "SCANNED" not in _build_document_context_note("scan.pdf", path, "application/pdf")


def test_probe_reads_only_the_opening_pages(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("subprocess.run", fake_run)
    _document_is_scanned_pdf(str(_pdf(tmp_path / "s.pdf", text=None)), "application/pdf")
    assert "-l" in seen["cmd"], "probe must bound the page range"
