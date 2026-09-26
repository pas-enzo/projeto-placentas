"""Gradio UI for the histomorphometry pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .capilar import CapilarModel
from .config import PipelineConfig
from .micro import MicrocotyledonModel
from .viz import colorize_overlay, masks_square_to_native

_CFG = PipelineConfig()
_MICRO: MicrocotyledonModel | None = None
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
        _MICRO = MicrocotyledonModel(_CFG, device=device)
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
    device: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if image is None:
        raise gr.Error("Envie uma imagem FOV (resolução nativa).")

    mode = (mode or "both").lower()
    if mode not in {"micro", "capilar", "both"}:
        raise gr.Error("Modo de análise inválido.")

    dev = None if device == "auto" else device
    _ensure_models(mode, dev)

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
            f"Microcotilédones — contagem: {micro.count} | "
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


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Histomorfometria Placentária Equina") as demo:
        gr.Markdown(
            """
# Histomorfometria Placentária Equina

Envie um **FOV nativo** para quantificar microcotilédones (RF-DETR) e/ou capilares (YOLO11s-seg + SAHI).

**Legenda do overlay:** verde = microcotilédones · ciano = capilares · contorno amarelo = contorno das instâncias
            """.strip()
        )

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
                with gr.Accordion("Avançado", open=False):
                    device = gr.Dropdown(
                        choices=[
                            ("Automático", "auto"),
                            ("GPU (CUDA)", "cuda:0"),
                            ("CPU", "cpu"),
                        ],
                        value="auto",
                        label="Dispositivo",
                    )
                btn = gr.Button("Rodar Inferência", variant="primary", size="lg")

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

        btn.click(
            fn=predict,
            inputs=[inp, mode, device],
            outputs=[out_orig, out_pred, out_txt],
        )
    return demo


def main() -> None:
    demo = build_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
