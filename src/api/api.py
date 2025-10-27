from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.api.schemas import PredictResponse, SampleItem
from src.api.samples import list_sample_paths, DEFAULT_STATIC_DIR, DEFAULT_SAMPLES_DIR, get_samples_dir
from src.api.deps import get_inference_service
from src.labels import index_to_label  # provided in your project context


app = FastAPI(
    title="TikkaMasalAI Food-101 API",
    description="Tiny API v1 for Food-101 inference with selectable sample images.",
    version="0.1.0",
)

# Mount static files only if the default static directory exists (for thumbnails)
if DEFAULT_STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DEFAULT_STATIC_DIR)), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root_ui():
    """
    Minimal UI so anyone can click a sample image and run inference without Postman.
    If you want a fancier frontend later, you can replace this.
    """
    samples = list_sample_paths(limit=10)
    if not samples:
        hint_dir = get_samples_dir().as_posix()
        return HTMLResponse(
            f"""
            <html>
            <head><title>Food101 Demo</title></head>
            <body style="font-family:system-ui;max-width:900px;margin:40px auto;">
              <h1>Food-101 Demo</h1>
              <p>Place 5–10 sample images in <code>{hint_dir}</code> (jpg/png/webp),
              then refresh this page.</p>
              <p>Try: <code>curl http://localhost:5000/health</code> or open <a href="/docs">/docs</a>.</p>
            </body>
            </html>
            """,
            status_code=200,
        )

    # Build cards
    cards = []
    for i, p in enumerate(samples, start=1):
        img_url = (
            f"/static/samples/{p.name}"
            if DEFAULT_SAMPLES_DIR.exists() and p.parent.resolve() == DEFAULT_SAMPLES_DIR.resolve()
            else ""  # no served image when using a custom external folder
        )
        thumb = f'<img src="{img_url}" alt="{p.name}" style="width:160px;height:160px;object-fit:cover;border-radius:12px;border:1px solid #eee;" />' if img_url else f"<code>{p.name}</code>"
        cards.append(
            f"""
            <div style="display:flex;flex-direction:column;gap:8px;align-items:center;padding:10px;border:1px solid #eee;border-radius:12px;">
              {thumb}
              <div style="font-size:12px;color:#666;max-width:160px;word-break:break-all;">{p.name}</div>
              <button onclick="predict({i})" style="border-radius:999px;padding:6px 12px;border:1px solid #ddd;background:#fafafa;cursor:pointer;">Predict</button>
            </div>
            """
        )

    grid = "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px;'>" + "".join(cards) + "</div>"

    return HTMLResponse(
        f"""
        <html>
        <head>
          <title>Food101 Demo</title>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
        </head>
        <body style="font-family:system-ui;max-width:1000px;margin:40px auto;padding:0 16px;">
          <h1>Food-101 Demo</h1>
          <p>Select one of the sample images to run inference.</p>
          {grid}
          <pre id="out" style="margin-top:24px;background:#0a0a0a;color:#e6e6e6;padding:16px;border-radius:12px;white-space:pre-wrap;"></pre>
          <script>
            async function predict(id) {{
              const res = await fetch(`/predict/${{id}}`);
              const json = await res.json();
              document.getElementById('out').textContent = JSON.stringify(json, null, 2);
              window.scrollTo({{ top: document.body.scrollHeight, behavior: 'smooth' }});
            }}
          </script>
          <p style="margin-top:24px;">API docs: <a href="/docs">/docs</a> • Redoc: <a href="/redoc">/redoc</a></p>
        </body>
        </html>
        """,
        status_code=200,
    )


@app.get("/samples", response_model=list[SampleItem])
def list_samples(limit: int = 10):
    """
    JSON endpoint listing up to N available samples. The `url` field works when
    using the default static folder (src/api/static/samples).
    """
    paths = list_sample_paths(limit=limit)
    items: list[SampleItem] = []
    for i, p in enumerate(paths, start=1):
        url = (
            f"/static/samples/{p.name}"
            if DEFAULT_SAMPLES_DIR.exists() and p.parent.resolve() == DEFAULT_SAMPLES_DIR.resolve()
            else f"file://{p.as_posix()}"
        )
        items.append(SampleItem(id=i, filename=p.name, url=url, label_hint=None))
    return items


@app.get("/predict/{sample_id}", response_model=PredictResponse)
def predict_sample(
    sample_id: int,
    svc=Depends(get_inference_service),
):
    """
    Run inference on one of the listed sample images by numeric id (1-based).
    """
    paths = list_sample_paths(limit=10)
    if sample_id < 1 or sample_id > len(paths):
        raise HTTPException(status_code=404, detail=f"sample_id {sample_id} not found")

    chosen = paths[sample_id - 1]
    image_bytes = chosen.read_bytes()

    idx = svc.predict(image_bytes)
    label = index_to_label(idx)

    return PredictResponse(
        sample_id=sample_id,
        predicted_index=idx,
        predicted_label=label,
        model_name=getattr(svc, "model_name", "unknown"),
        bytes_read=len(image_bytes),
    )
