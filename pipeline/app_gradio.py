"""Gradio UI for the histomorphometry pipeline."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

_APP_OUT = Path(__file__).resolve().parents[1] / "outputs" / "app"
# Gradio reads its upload/cache dir at import time; keep it on D: (C: is nearly full).
os.environ.setdefault("GRADIO_TEMP_DIR", str(_APP_OUT / "gradio_cache"))

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from .capilar import CapilarModel
from .config import PipelineConfig
from .micro import MicrocotyledonModel, MicrocotyledonYOLOModel, load_micro_model
from .run import analyze_image, write_results_csv
from .types import ImageResult
from .viz import colorize_overlay, masks_square_to_native, save_side_by_side

_CFG = PipelineConfig()
_MICRO: MicrocotyledonModel | MicrocotyledonYOLOModel | None = None
_CAP: CapilarModel | None = None

_MODE_CHOICES = [
    ("Microcotilédones + Capilares", "both"),
    ("Só Microcotilédones", "micro"),
    ("Só Capilares", "capilar"),
]
_MODE_LABEL = {v: k for k, v in _MODE_CHOICES}


def _ensure_models(mode: str, device: str | None) -> None:
    global _MICRO, _CAP
    mode = mode.lower()
    if mode in {"micro", "both"} and _MICRO is None:
        _MICRO = load_micro_model(_CFG, device=device)
    if mode in {"capilar", "both"} and _CAP is None:
        _CAP = CapilarModel(_CFG, device=device)


def _bgr_from_upload(image: np.ndarray) -> np.ndarray:
    """Gradio Image(type='numpy') is RGB; pipeline expects BGR."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict(
    image: np.ndarray | None,
    mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if image is None:
        raise gr.Error("Envie uma imagem FOV (resolução nativa).")

    mode = (mode or "both").lower()
    if mode not in {"micro", "capilar", "both"}:
        raise gr.Error("Modo de análise inválido.")

    # Always auto: CUDA if available, otherwise CPU (set inside model loaders).
    _ensure_models(mode, device=None)

    bgr = _bgr_from_upload(image)
    h, w = bgr.shape[:2]
    micro_native = None
    mode_label = _MODE_LABEL.get(mode, mode)
    lines: list[str] = [
        f"FOV: {w} × {h} px",
        f"Análise: {mode_label}",
    ]

    if mode in {"micro", "both"}:
        assert _MICRO is not None
        micro = _MICRO.predict_bgr(bgr)
        micro_native = masks_square_to_native(
            micro.masks, native_w=w, native_h=h, canvas=_CFG.micro_canvas
        )
        lines.append(
            f"Microcotilédones ({micro.model_name}) — contagem: {micro.count} | "
            f"área: {micro.area_um2:,.2f} µm² ({micro.area_px:,.0f} px)"
        )

    cap_masks = None
    if mode in {"capilar", "both"}:
        assert _CAP is not None
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), bgr)
            cap = _CAP.predict_path(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        cap_masks = cap.masks
        lines.append(
            f"Capilares — contagem: {cap.count} | "
            f"área: {cap.area_um2:,.2f} µm² ({cap.area_px:,.0f} px)"
        )

    overlay_bgr = colorize_overlay(
        bgr, micro_masks=micro_native, capilar_masks=cap_masks
    )
    return _to_rgb(bgr), _to_rgb(overlay_bgr), "\n".join(lines)


def _batch_caption(res: ImageResult) -> str:
    parts = [res.image_path.name]
    if res.micro is not None:
        parts.append(f"Micro: {res.micro.count} | {res.micro.area_um2:,.0f} µm²")
    if res.capilar is not None:
        parts.append(f"Capilares: {res.capilar.count} | {res.capilar.area_um2:,.0f} µm²")
    return "\n".join(parts)


def _batch_table(results: list[ImageResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"Imagem": r.image_path.name}
        if r.micro is not None:
            row["Microcotilédones"] = r.micro.count
            row["Área micro (µm²)"] = round(r.micro.area_um2, 2)
        if r.capilar is not None:
            row["Capilares"] = r.capilar.count
            row["Área capilares (µm²)"] = round(r.capilar.area_um2, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def predict_batch(
    files: list[str] | None,
    mode: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[list[tuple[str, str]], pd.DataFrame, list[str], str]:
    if not files:
        raise gr.Error("Envie uma ou mais imagens FOV (resolução nativa).")

    mode = (mode or "both").lower()
    if mode not in {"micro", "capilar", "both"}:
        raise gr.Error("Modo de análise inválido.")

    progress(0, desc="Carregando modelos...")
    _ensure_models(mode, device=None)
    micro_model = _MICRO if mode in {"micro", "both"} else None
    cap_model = _CAP if mode in {"capilar", "both"} else None

    out_dir = _APP_OUT / "lote" / time.strftime("placentas_lote_%Y%m%d_%H%M%S")
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True)

    t0 = time.perf_counter()
    results: list[ImageResult] = []
    gallery: list[tuple[str, str]] = []
    for f in progress.tqdm(files, desc="Processando imagens"):
        img_path = Path(f)
        res, bgr, overlay = analyze_image(
            img_path, cfg=_CFG, micro_model=micro_model, cap_model=cap_model
        )
        overlay_path = overlay_dir / f"{img_path.stem}_overlay.jpg"
        save_side_by_side(bgr, overlay, overlay_path)
        res.overlay_path = Path("overlays") / overlay_path.name
        res.drop_masks()
        results.append(res)
        gallery.append((str(overlay_path), _batch_caption(res)))
    elapsed = time.perf_counter() - t0

    csv_path = out_dir / "results.csv"
    write_results_csv(results, csv_path)
    zip_path = Path(shutil.make_archive(str(out_dir.parent / out_dir.name), "zip", out_dir))

    n = len(results)
    lines = [
        f"Imagens: {n} | Análise: {_MODE_LABEL.get(mode, mode)} | "
        f"Tempo: {elapsed:.1f} s ({elapsed / n:.1f} s/imagem)",
    ]
    if micro_model is not None:
        total = sum(r.micro.area_um2 for r in results if r.micro)
        count = sum(r.micro.count for r in results if r.micro)
        lines.append(
            f"Microcotilédones ({results[0].micro.model_name}) — total: {count} | "
            f"área média por campo: {total / n:,.2f} µm²"
        )
    if cap_model is not None:
        total = sum(r.capilar.area_um2 for r in results if r.capilar)
        count = sum(r.capilar.count for r in results if r.capilar)
        lines.append(
            f"Capilares — total: {count} | área média por campo: {total / n:,.2f} µm²"
        )
    return gallery, _batch_table(results), [str(csv_path), str(zip_path)], "\n".join(lines)


def _legend_html() -> str:
    """Compact swatches; vertically aligned with label text."""
    return """
<div style="display:flex;flex-wrap:wrap;gap:14px 22px;align-items:center;
            line-height:1.2;margin:6px 0 10px 0;color:inherit;">
  <strong style="display:inline-flex;align-items:center;height:1.2em;">Legenda</strong>
  <span style="display:inline-flex;align-items:center;gap:8px;height:1.2em;">
    <span style="width:20px;height:12px;border-radius:2px;background:#00ff00;
                 border:2px solid #ffff00;display:inline-block;box-sizing:border-box;
                 flex-shrink:0;"></span>
    <span style="display:inline-flex;align-items:center;">Microcotilédones</span>
  </span>
  <span style="display:inline-flex;align-items:center;gap:8px;height:1.2em;">
    <span style="width:20px;height:12px;border-radius:2px;background:#00ffff;
                 border:2px solid #ffff00;display:inline-block;box-sizing:border-box;
                 flex-shrink:0;"></span>
    <span style="display:inline-flex;align-items:center;">Capilares</span>
  </span>
</div>
""".strip()


def build_app() -> gr.Blocks:
    # Only background + primary accents (replaces default orange).
    theme = gr.themes.Default().set(
        body_background_fill="#1e1f22",
        body_background_fill_dark="#1e1f22",
        button_primary_background_fill="#5E6DBA",
        button_primary_background_fill_hover="#4F5DA8",
        button_primary_background_fill_dark="#5E6DBA",
        button_primary_background_fill_hover_dark="#4F5DA8",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        # Radio selected + loading spinner (Gradio accent).
        checkbox_background_color_selected="#5E6DBA",
        checkbox_background_color_selected_dark="#5E6DBA",
        checkbox_border_color_selected="#5E6DBA",
        checkbox_border_color_selected_dark="#5E6DBA",
        color_accent="#5E6DBA",
        color_accent_soft="#5E6DBA",
        color_accent_soft_dark="#5E6DBA",
        border_color_accent="#5E6DBA",
        border_color_accent_dark="#5E6DBA",
        border_color_accent_subdued="#5E6DBA",
        border_color_accent_subdued_dark="#5E6DBA",
    )

    with gr.Blocks(
        title="Histomorfometria Placentária Equina",
        theme=theme,
    ) as demo:
        # Legend lives in the same Markdown block so left edge matches the title.
        gr.Markdown(
            f"""
# Histomorfometria Placentária Equina

Envie um **FOV nativo** para quantificar microcotilédones e/ou capilares (YOLO11s-seg + SAHI).

{_legend_html()}
            """.strip()
        )

        with gr.Tab("Imagem Única"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### Entrada")
                    inp = gr.Image(
                        type="numpy",
                        label="FOV nativo",
                        sources=["upload"],
                        height=360,
                    )
                    mode = gr.Radio(
                        choices=_MODE_CHOICES,
                        value="both",
                        label="Tipo de análise",
                    )
                    with gr.Row():
                        btn = gr.Button("Rodar Inferência", variant="primary", scale=2)
                        btn_clear = gr.Button("Limpar", variant="secondary", scale=1)

                with gr.Column(scale=2, min_width=480):
                    gr.Markdown("### Resultados")
                    with gr.Row():
                        out_orig = gr.Image(
                            type="numpy",
                            label="Original",
                            interactive=False,
                            height=360,
                        )
                        out_pred = gr.Image(
                            type="numpy",
                            label="Predição (overlay)",
                            interactive=False,
                            height=360,
                        )
                    out_txt = gr.Textbox(
                        label="Métricas",
                        lines=6,
                        interactive=False,
                    )

        with gr.Tab("Lote"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### Entrada")
                    batch_files = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                        label="FOVs nativos",
                        height=360,
                    )
                    batch_mode = gr.Radio(
                        choices=_MODE_CHOICES,
                        value="both",
                        label="Tipo de análise",
                    )
                    with gr.Row():
                        batch_btn = gr.Button("Rodar Lote", variant="primary", scale=2)
                        batch_clear = gr.Button("Limpar", variant="secondary", scale=1)

                with gr.Column(scale=2, min_width=480):
                    gr.Markdown("### Resultados")
                    batch_gallery = gr.Gallery(
                        label="Overlays (original | predição)",
                        columns=2,
                        height=360,
                        object_fit="contain",
                        interactive=False,
                    )
                    batch_txt = gr.Textbox(
                        label="Resumo",
                        lines=3,
                        interactive=False,
                    )
                    batch_table = gr.Dataframe(
                        label="Métricas por imagem",
                        interactive=False,
                        wrap=True,
                    )
                    batch_download = gr.File(
                        label="Download (results.csv + ZIP com overlays)",
                        file_count="multiple",
                        interactive=False,
                    )

        btn.click(
            fn=predict,
            inputs=[inp, mode],
            outputs=[out_orig, out_pred, out_txt],
        )

        def _clear():
            return None, None, None, ""

        btn_clear.click(
            fn=_clear,
            inputs=None,
            outputs=[inp, out_orig, out_pred, out_txt],
        )

        batch_btn.click(
            fn=predict_batch,
            inputs=[batch_files, batch_mode],
            outputs=[batch_gallery, batch_table, batch_download, batch_txt],
        )

        def _clear_batch():
            return None, None, None, None, ""

        batch_clear.click(
            fn=_clear_batch,
            inputs=None,
            outputs=[batch_files, batch_gallery, batch_table, batch_download, batch_txt],
        )
    return demo


def main() -> None:
    demo = build_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
