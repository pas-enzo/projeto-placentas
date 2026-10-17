"""Gradio UI for the histomorphometry pipeline."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

_APP_OUT = Path(__file__).resolve().parents[1] / "outputs" / "app"
# Gradio reads its upload/cache dir at import time; keep it on D: (C: is nearly full).
os.environ.setdefault("GRADIO_TEMP_DIR", str(_APP_OUT / "gradio_cache"))

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from .artifacts import new_run_dir, save_image_artifacts, unique_stem, write_run_artifacts
from .capilar import CapilarModel
from .config import (
    NATIVE_H,
    NATIVE_W,
    SCALE_BAR_PX_NATIVE,
    SCALE_BAR_UM,
    UM_PER_PX_NATIVE,
    PipelineConfig,
    fov_size_warning,
)
from .interactive import HOVER_CSS, HOVER_HEAD, interactive_overlay_html
from .micro import MicrocotyledonModel, MicrocotyledonYOLOModel, load_micro_model
from .run import analyze_bgr, analyze_image
from .stats import batch_table, image_table
from .types import ImageResult

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


def _finish_run(results: list[ImageResult], run_dir: Path, mode: str) -> list[str]:
    """Write the run-level files and zip the folder; same output for both tabs."""
    paths = write_run_artifacts(results, run_dir, _CFG, mode=mode)
    zip_path = Path(shutil.make_archive(str(run_dir.parent / run_dir.name), "zip", run_dir))
    return [str(zip_path), str(paths["results"]), str(paths["instances"]), str(paths["coco"])]


def _zip_button(zip_path: str | None) -> gr.DownloadButton:
    return gr.DownloadButton("Baixar tudo (ZIP)", value=zip_path, interactive=zip_path is not None)


def predict(
    image: np.ndarray | None,
    mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str, pd.DataFrame, gr.DownloadButton, list[str], str]:
    if image is None:
        raise gr.Error("Envie uma imagem FOV (resolução nativa).")

    mode = (mode or "both").lower()
    if mode not in {"micro", "capilar", "both"}:
        raise gr.Error("Modo de análise inválido.")

    # Always auto: CUDA if available, otherwise CPU (set inside model loaders).
    _ensure_models(mode, device=None)

    bgr = _bgr_from_upload(image)
    h, w = bgr.shape[:2]
    mode_label = _MODE_LABEL.get(mode, mode)
    lines: list[str] = [
        f"FOV: {w} × {h} px",
        f"Análise: {mode_label}",
    ]
    fov_warning = fov_size_warning(w, h)
    if fov_warning:
        gr.Warning(fov_warning)
        lines.append(f"AVISO: {fov_warning}")

    run_dir = new_run_dir(_APP_OUT / "unica", "placentas_unica")
    res, bgr, overlay = analyze_bgr(
        bgr,
        Path("imagem_unica.png"),
        cfg=_CFG,
        micro_model=_MICRO if mode in {"micro", "both"} else None,
        cap_model=_CAP if mode in {"capilar", "both"} else None,
        from_path=False,
    )
    hover_html = interactive_overlay_html(
        bgr, micro=res.micro, capilar=res.capilar, micro_canvas=_CFG.micro_canvas
    )
    save_image_artifacts(res, bgr, overlay, run_dir, "imagem_unica", _CFG)
    header, rows = image_table(res, _CFG)
    downloads = _finish_run([res], run_dir, mode)
    return (
        _to_rgb(bgr),
        _to_rgb(overlay),
        "\n".join(lines),
        pd.DataFrame(rows, columns=header),
        _zip_button(downloads[0]),
        downloads[1:],
        hover_html,
    )


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
) -> tuple[list[tuple[str, str]], pd.DataFrame, pd.DataFrame, gr.DownloadButton, list[str], str]:
    if not files:
        raise gr.Error("Envie uma ou mais imagens FOV (resolução nativa).")

    mode = (mode or "both").lower()
    if mode not in {"micro", "capilar", "both"}:
        raise gr.Error("Modo de análise inválido.")

    progress(0, desc="Carregando modelos...")
    _ensure_models(mode, device=None)
    micro_model = _MICRO if mode in {"micro", "both"} else None
    cap_model = _CAP if mode in {"capilar", "both"} else None

    out_dir = new_run_dir(_APP_OUT / "lote", "placentas_lote")
    used_names: set[str] = set()

    t0 = time.perf_counter()
    results: list[ImageResult] = []
    gallery: list[tuple[str, str]] = []
    for f in progress.tqdm(files, desc="Processando imagens"):
        img_path = Path(f)
        res, bgr, overlay = analyze_image(
            img_path, cfg=_CFG, micro_model=micro_model, cap_model=cap_model
        )
        preview = save_image_artifacts(
            res, bgr, overlay, out_dir, unique_stem(img_path.stem, used_names), _CFG
        )
        res.drop_masks()
        results.append(res)
        gallery.append((str(preview), _batch_caption(res)))
    elapsed = time.perf_counter() - t0

    downloads = _finish_run(results, out_dir, mode)

    n = len(results)
    lines = [
        f"Imagens: {n} | Análise: {_MODE_LABEL.get(mode, mode)} | "
        f"Tempo: {elapsed:.1f} s ({elapsed / n:.1f} s/imagem)",
    ]
    lines.append(
        "Métricas do lote: contagem, áreas, sobreposição e ocupação = média ± dp por campo; "
        "estatísticas por instância = todas as instâncias do lote."
    )
    n_warn = sum(1 for r in results if r.extras.get("warning"))
    if n_warn:
        msg = f"{n_warn} imagem(ns) com tamanho diferente de {NATIVE_W}×{NATIVE_H} px: " \
            "área em µm² pode estar incorreta (coluna 'warning' do CSV)."
        gr.Warning(msg)
        lines.append(f"AVISO: {msg}")
    header, rows = batch_table(results, _CFG)
    return (
        gallery,
        pd.DataFrame(rows, columns=header),
        _batch_table(results),
        _zip_button(downloads[0]),
        downloads[1:],
        "\n".join(lines),
    )


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


def _area_factors_md(cfg: PipelineConfig) -> str:
    """Short explanation of the area conversion; numbers come straight from the config."""
    return f"""
- **Calibração:** a barra de escala de {SCALE_BAR_UM:g} µm mede {SCALE_BAR_PX_NATIVE:g} px no FOV nativo ({NATIVE_W} × {NATIVE_H} px), ou seja, {UM_PER_PX_NATIVE:.4f} µm por pixel.
- **Capilares** (pixel nativo): {cfg.capilar_area_um2_per_px2:.5f} µm² por px².
- **Microcotilédones** (imagem esticada para {cfg.micro_canvas} × {cfg.micro_canvas} px, cada pixel cobre {NATIVE_W / cfg.micro_canvas:.2f} × {NATIVE_H / cfg.micro_canvas:.2f} px nativos): {cfg.micro_area_um2_per_px2:.4f} µm² por px².
- **Área total** = soma das áreas das instâncias; a **união** desconta as sobreposições entre máscaras.
- Os valores em µm² só valem para FOVs nativos de {NATIVE_W} × {NATIVE_H} px.
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
        css=HOVER_CSS,
        head=HOVER_HEAD,
    ) as demo:
        # Legend lives in the same Markdown block so left edge matches the title.
        gr.Markdown(
            f"""
# Histomorfometria Placentária Equina

Envie um **FOV nativo** (4140 × 3096 px) para quantificar microcotilédones e/ou capilares.

{_legend_html()}
            """.strip()
        )
        with gr.Accordion("Como a área é calculada?", open=False):
            gr.Markdown(_area_factors_md(_CFG))

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
                        label="Informações",
                        lines=3,
                        interactive=False,
                    )
                    out_table = gr.Dataframe(
                        label="Métricas",
                        interactive=False,
                        wrap=True,
                    )
                    out_zip = gr.DownloadButton("Baixar tudo (ZIP)", variant="primary", interactive=False)
                    out_download = gr.File(
                        label="Arquivos individuais (results.csv, instances.csv, annotations.coco.json)",
                        file_count="multiple",
                        interactive=False,
                    )
                    gr.Markdown("### Predição interativa")
                    out_hover = gr.HTML()

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
                        label="Informações",
                        lines=3,
                        interactive=False,
                    )
                    batch_summary = gr.Dataframe(
                        label="Métricas do lote",
                        interactive=False,
                        wrap=True,
                    )
                    batch_table = gr.Dataframe(
                        label="Métricas por imagem",
                        interactive=False,
                        wrap=True,
                    )
                    batch_zip = gr.DownloadButton("Baixar tudo (ZIP)", variant="primary", interactive=False)
                    batch_download = gr.File(
                        label="Arquivos individuais (results.csv, instances.csv, annotations.coco.json)",
                        file_count="multiple",
                        interactive=False,
                    )

        btn.click(
            fn=predict,
            inputs=[inp, mode],
            outputs=[out_orig, out_pred, out_txt, out_table, out_zip, out_download, out_hover],
        )

        def _clear():
            return None, None, None, "", None, _zip_button(None), None, ""

        btn_clear.click(
            fn=_clear,
            inputs=None,
            outputs=[inp, out_orig, out_pred, out_txt, out_table, out_zip, out_download, out_hover],
        )

        batch_btn.click(
            fn=predict_batch,
            inputs=[batch_files, batch_mode],
            outputs=[batch_gallery, batch_summary, batch_table, batch_zip, batch_download, batch_txt],
        )

        def _clear_batch():
            return None, None, None, None, _zip_button(None), None, ""

        batch_clear.click(
            fn=_clear_batch,
            inputs=None,
            outputs=[batch_files, batch_gallery, batch_summary, batch_table, batch_zip, batch_download, batch_txt],
        )
    return demo


def main() -> None:
    demo = build_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
