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
from .viz import colorize_overlay, masks_square_to_native, save_side_by_side

_CFG = PipelineConfig()
_MICRO: MicrocotyledonModel | None = None
_CAP: CapilarModel | None = None


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


def predict(
    image: np.ndarray | None,
    mode: str,
    device: str,
) -> tuple[np.ndarray | None, str]:
    if image is None:
        raise gr.Error("Envie uma imagem FOV (nativa).")

    mode = (mode or "both").lower()
    if mode not in {"micro", "capilar", "both"}:
        raise gr.Error("Mode inválido.")

    dev = None if device == "auto" else device
    _ensure_models(mode, dev)

    bgr = _bgr_from_upload(image)
    h, w = bgr.shape[:2]
    micro_native = None
    lines: list[str] = [f"FOV: {w}×{h} px", f"Mode: {mode}"]

    if mode in {"micro", "both"}:
        assert _MICRO is not None
        micro = _MICRO.predict_bgr(bgr)
        micro_native = masks_square_to_native(
            micro.masks, native_w=w, native_h=h, canvas=_CFG.micro_canvas
        )
        lines.append(
            f"Microcotilédones: count={micro.count} | "
            f"área={micro.area_um2:.2f} µm² ({micro.area_px:.0f} px)"
        )

    cap_masks = None
    if mode in {"capilar", "both"}:
        assert _CAP is not None
        # SAHI API expects a path.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), bgr)
            cap = _CAP.predict_path(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        cap_masks = cap.masks
        lines.append(
            f"Capilares: count={cap.count} | "
            f"área={cap.area_um2:.2f} µm² ({cap.area_px:.0f} px)"
        )

    overlay = colorize_overlay(bgr, micro_masks=micro_native, capilar_masks=cap_masks)
    with tempfile.NamedTemporaryFile(suffix="_overlay.jpg", delete=False) as tmp:
        side_path = Path(tmp.name)
    save_side_by_side(bgr, overlay, side_path, max_w=2400)
    side_bgr = cv2.imread(str(side_path), cv2.IMREAD_COLOR)
    side_path.unlink(missing_ok=True)
    if side_bgr is None:
        raise gr.Error("Falha ao montar overlay.")
    side_rgb = cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB)
    return side_rgb, "\n".join(lines)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Placentas — histomorfometria") as demo:
        gr.Markdown(
            "# Histomorfometria placentária equina\n"
            "Upload de FOV nativo → microcotilédones (RF-DETR) e/ou capilares (YOLO+SAHI).\n\n"
            "Overlay: **verde** = micro · **ciano** = capilar (ambos com contorno amarelo)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="numpy", label="FOV (imagem nativa)")
                mode = gr.Radio(
                    choices=["both", "micro", "capilar"],
                    value="both",
                    label="Mode",
                )
                device = gr.Dropdown(
                    choices=["auto", "cuda:0", "cpu"],
                    value="auto",
                    label="Device",
                )
                btn = gr.Button("Rodar inferência", variant="primary")
            with gr.Column(scale=2):
                out_img = gr.Image(type="numpy", label="Original | Predição")
                out_txt = gr.Textbox(label="Métricas", lines=6)

        btn.click(fn=predict, inputs=[inp, mode, device], outputs=[out_img, out_txt])
    return demo


def main() -> None:
    demo = build_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
