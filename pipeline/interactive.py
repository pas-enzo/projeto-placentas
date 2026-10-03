"""Hoverable SVG overlay: one path per instance with its area in a tooltip."""

from __future__ import annotations

import base64
import html

import cv2
import numpy as np

from .types import InstanceResult

HOVER_CSS = """
.hv-wrap { position: relative; width: 100%; }
.hv-hint { font-size: 0.85em; opacity: 0.75; margin: 0 0 6px 0; }
.hv-svg { width: 100%; height: auto; display: block; background: #000;
          cursor: crosshair; user-select: none; touch-action: none; }
.hv-svg.hv-drag { cursor: grabbing; }
.hv-inst { fill-rule: evenodd; vector-effect: non-scaling-stroke;
           transition: fill-opacity 0.08s; }
.hv-micro { fill: #00ff00; fill-opacity: 0.30; stroke: #ffff00; stroke-width: 1.5; }
.hv-cap { fill: #00ffff; fill-opacity: 0.45; stroke: #ffff00; stroke-width: 1; }
.hv-inst:hover { fill-opacity: 0.85; stroke: #ffffff; stroke-width: 3; }
.hv-tip { position: absolute; display: none; pointer-events: none; z-index: 10;
          white-space: pre; font-size: 13px; line-height: 1.35; color: #fff;
          background: rgba(20, 20, 24, 0.92); border: 1px solid #5E6DBA;
          border-radius: 4px; padding: 6px 8px; }
"""

# Event delegation on document: gr.HTML re-renders its innerHTML on every update.
HOVER_HEAD = """
<script>
(() => {
  if (window.__hvInit) return;
  window.__hvInit = true;
  let tip = null, drag = null;

  const hideTip = () => { if (tip) { tip.style.display = "none"; tip = null; } };
  const svgPoint = (svg, e) => {
    const v = svg.viewBox.baseVal, r = svg.getBoundingClientRect();
    return [v.x + (e.clientX - r.left) / r.width * v.width,
            v.y + (e.clientY - r.top) / r.height * v.height];
  };
  const setView = (svg, x, y, w) => {
    const W = +svg.dataset.w, H = +svg.dataset.h;
    w = Math.min(W, Math.max(W / 12, w));
    const h = w * H / W;
    x = Math.max(0, Math.min(W - w, x));
    y = Math.max(0, Math.min(H - h, y));
    svg.setAttribute("viewBox", `${x} ${y} ${w} ${h}`);
  };

  document.addEventListener("mousemove", (e) => {
    if (drag) {
      const v = drag.svg.viewBox.baseVal, r = drag.svg.getBoundingClientRect();
      const k = v.width / r.width;
      setView(drag.svg, drag.x - (e.clientX - drag.cx) * k,
              drag.y - (e.clientY - drag.cy) * k, v.width);
      hideTip();
      return;
    }
    const inst = e.target.closest ? e.target.closest(".hv-inst") : null;
    if (!inst) { hideTip(); return; }
    const wrap = inst.closest(".hv-wrap");
    const t = wrap.querySelector(".hv-tip");
    if (tip && tip !== t) hideTip();
    tip = t;
    t.textContent = inst.dataset.tip;
    t.style.display = "block";
    const r = wrap.getBoundingClientRect();
    let x = e.clientX - r.left + 14, y = e.clientY - r.top + 14;
    if (x + t.offsetWidth > r.width) x -= t.offsetWidth + 28;
    if (y + t.offsetHeight > r.height) y -= t.offsetHeight + 28;
    t.style.left = x + "px";
    t.style.top = y + "px";
  });

  document.addEventListener("wheel", (e) => {
    const svg = e.target.closest ? e.target.closest(".hv-svg") : null;
    if (!svg) return;
    e.preventDefault();
    const v = svg.viewBox.baseVal;
    const [px, py] = svgPoint(svg, e);
    const w = Math.min(+svg.dataset.w, Math.max(+svg.dataset.w / 12,
                       v.width * (e.deltaY < 0 ? 0.8 : 1.25)));
    const s = w / v.width;
    setView(svg, px - (px - v.x) * s, py - (py - v.y) * s, w);
  }, { passive: false });

  document.addEventListener("mousedown", (e) => {
    const svg = e.target.closest ? e.target.closest(".hv-svg") : null;
    if (!svg || e.button !== 0) return;
    e.preventDefault();
    const v = svg.viewBox.baseVal;
    drag = { svg, cx: e.clientX, cy: e.clientY, x: v.x, y: v.y };
    svg.classList.add("hv-drag");
  });

  document.addEventListener("mouseup", () => {
    if (drag) { drag.svg.classList.remove("hv-drag"); drag = null; }
  });

  document.addEventListener("dblclick", (e) => {
    const svg = e.target.closest ? e.target.closest(".hv-svg") : null;
    if (svg) setView(svg, 0, 0, +svg.dataset.w);
  });
})();
</script>
"""


def _path_d(
    mask: np.ndarray, sx: float, sy: float, ox: int = 0, oy: int = 0, eps: float = 0.5
) -> str:
    """SVG path (outer contours + holes) scaled from mask pixels to display units."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    parts: list[str] = []
    for c in contours:
        c = cv2.approxPolyDP(c, eps, True)
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        xs = (pts[:, 0] + ox + 0.5) * sx
        ys = (pts[:, 1] + oy + 0.5) * sy
        parts.append("M" + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + "Z")
    return "".join(parts)


def _crop_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _inst_path(cls: str, d: str, tip: str) -> str:
    return (
        f'<path class="hv-inst {cls}" d="{d}" '
        f'data-tip="{html.escape(tip, quote=True)}"/>'
    )


def _tip(label: str, idx: int, area: float, conf: float | None) -> str:
    lines = [f"{label} #{idx}", f"Área: {area:,.2f} µm²"]
    if conf is not None:
        lines.append(f"Confiança: {conf:.2f}")
    return "\n".join(lines)


def interactive_overlay_html(
    bgr: np.ndarray,
    micro: InstanceResult | None = None,
    capilar: InstanceResult | None = None,
    *,
    micro_canvas: int = 640,
    max_w: int = 2400,
) -> str:
    """Build the hover/zoom viewer. Micro masks are on the stretch canvas; capillary masks are native."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_w / w)
    dw, dh = int(round(w * scale)), int(round(h * scale))
    disp = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    ok, jpg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Could not encode display image")
    b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

    micro_paths: list[str] = []
    if micro is not None:
        sx, sy = dw / micro_canvas, dh / micro_canvas
        for i, m in enumerate(micro.masks):
            d = _path_d(m, sx, sy)
            if not d:
                continue
            conf = micro.confidences[i] if i < len(micro.confidences) else None
            micro_paths.append(
                _inst_path("hv-micro", d, _tip("Microcotilédone", i + 1, micro.areas_um2[i], conf))
            )

    cap_paths: list[str] = []
    if capilar is not None:
        for i, m in enumerate(capilar.masks):
            box = _crop_box(m)
            if box is None:
                continue
            y0, y1, x0, x1 = box
            d = _path_d(m[y0:y1, x0:x1], scale, scale, ox=x0, oy=y0, eps=1.0)
            if not d:
                continue
            conf = capilar.confidences[i] if i < len(capilar.confidences) else None
            cap_paths.append(
                _inst_path("hv-cap", d, _tip("Capilar", i + 1, capilar.areas_um2[i], conf))
            )

    return (
        '<div class="hv-wrap">'
        '<p class="hv-hint">Passe o mouse sobre uma instância para ver a área · '
        "roda do mouse: zoom · arrastar: mover · duplo clique: enquadrar</p>"
        f'<svg class="hv-svg" viewBox="0 0 {dw} {dh}" data-w="{dw}" data-h="{dh}" '
        f'style="aspect-ratio: {dw} / {dh};" xmlns="http://www.w3.org/2000/svg">'
        f'<image href="data:image/jpeg;base64,{b64}" x="0" y="0" width="{dw}" height="{dh}"/>'
        f'<g>{"".join(micro_paths)}</g>'
        f'<g>{"".join(cap_paths)}</g>'
        "</svg>"
        '<div class="hv-tip"></div>'
        "</div>"
    )
