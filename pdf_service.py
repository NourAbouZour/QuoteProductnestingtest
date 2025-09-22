# FastAPI microservice exposing one POST /generate endpoint
# Inputs: JSON payload (same shape as example_payload.json)
# Outputs: streams back the PDF and XLSX as a ZIP (or JSON paths if you prefer)

import io
import json
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from doc_generator import DocGenerator

APP_DIR = Path(__file__).parent.resolve()
TEMPLATE_XLSX = APP_DIR / "quote_template.xlsx"
CELL_MAP_JSON = APP_DIR / "quote_cell_map.json"
LOGO_DEFAULT = (APP_DIR / "assets" / "logo.jpg")

# Load mapping once
CELL_MAP = json.loads(CELL_MAP_JSON.read_text())

app = FastAPI(title="Quotation PDF/XLSX Generator")

class GenerateQuery(BaseModel):
    title: Optional[str] = "Quotation"
    logo_path: Optional[str] = None  # default to assets/logo.png if present

@app.post("/generate")
def generate(
    payload: dict = Body(..., description="Quotation payload JSON"),
    title: str = Query("Quotation"),
    logo_path: Optional[str] = Query(None),
    return_zip: bool = Query(True, description="Return a ZIP of PDF+XLSX")
):
    # Validate existence
    if not TEMPLATE_XLSX.exists():
        raise HTTPException(500, "quote_template.xlsx not found next to pdf_service.py")
    gen = DocGenerator(excel_template=str(TEMPLATE_XLSX), cell_map=CELL_MAP)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        pdf_path = out_dir / "quote.pdf"
        xlsx_path = out_dir / "quote.xlsx"

        # Render PDF
        gen.render_pdf(
            payload, str(pdf_path),
            title=title,
            logo_path=str(LOGO_DEFAULT) if (not logo_path and LOGO_DEFAULT.exists()) else logo_path
        )

        # Fill Excel
        gen.fill_excel(payload, str(xlsx_path))

        if not return_zip:
            # Return JSON with base64? or temp paths? For simplicity, enforce ZIP by default.
            return {"pdf": str(pdf_path), "xlsx": str(xlsx_path)}

        # Stream a zip containing both files
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pdf_path, arcname="quote.pdf")
            z.write(xlsx_path, arcname="quote.xlsx")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="quote_bundle.zip"'}
        )
# ---- Optional helpers for Flask apps (importable from app.py) ----
from pathlib import Path as _Path
import io as _io
import json as _json
import tempfile as _tempfile
from doc_generator import DocGenerator as _DocGenerator

def get_placeholder_fields():
    """
    Return the dotted keys expected by the Excel mapping.
    Use this to pre-build your form or sanity check payloads.
    """
    here = _Path(__file__).parent.resolve()
    cell_map = _json.loads((here / "quote_cell_map.json").read_text())
    return sorted([k for k in cell_map.keys() if not k.endswith("_anchor")])

def generate_pdf(payload: dict, *, title: str = "Quotation", logo_path: str | None = None):
    """
    Generate PDF + XLSX in-memory and return (pdf_bytes, xlsx_bytes).
    Designed to be used from Flask routes.
    """
    here = _Path(__file__).parent.resolve()
    template = here / "quote_template.xlsx"
    cell_map = _json.loads((here / "quote_cell_map.json").read_text())
    gen = _DocGenerator(excel_template=str(template), cell_map=cell_map)

    with _tempfile.TemporaryDirectory() as tmpdir:
        out_pdf = _Path(tmpdir) / "quote.pdf"
        out_xlsx = _Path(tmpdir) / "quote.xlsx"

        gen.render_pdf(payload, str(out_pdf), title=title, logo_path=logo_path)
        gen.fill_excel(payload, str(out_xlsx))

        return out_pdf.read_bytes(), out_xlsx.read_bytes()
