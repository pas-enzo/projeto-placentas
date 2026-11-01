"""Mask R-CNN family (torchvision), trained and evaluated through the common benchmark interface.

Models: `maskrcnn_resnet50_fpn` (v1) and `maskrcnn_resnet50_fpn_v2`, both starting from COCO weights.
Data: the Roboflow-style COCO export of the view (`<fold>/coco/<view>/{train,valid,test}`).

Training choices (identical for every model of this family, see docs/BENCHMARK_PLAN.md):
  * SGD (momentum 0.9, weight decay 1e-4), warm-up then cosine decay; FP32 (AMP is unreliable on the GTX 1650 SUPER).
  * Augmentation: random flips and 90-degree rotations (the orientation-free analogue of the YOLO recipes).
  * No early stopping: the weights of the LAST epoch are used (the cosine schedule ends at a very low LR).
`supports_resume = True`: `last.pt` (model, optimizer, scheduler, epoch) is written atomically after every epoch.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from benchmark.core import RunSpec, view_dir, write_json
from benchmark.eval.instances import Inst, from_full

os.environ.setdefault("TORCH_HOME", r"D:\projeto_placentas_clayton\weights_torchvision")  # keep downloads off C:

BUILDERS = {"maskrcnn_resnet50_fpn": "MaskRCNN_ResNet50_FPN_Weights", "maskrcnn_resnet50_fpn_v2": "MaskRCNN_ResNet50_FPN_V2_Weights"}


def build_model(name: str, imgsz: int, pretrained: bool = True, score_thresh: float = 0.05):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if name not in BUILDERS:
        raise KeyError(f"modelo torchvision desconhecido {name!r}; opções: {list(BUILDERS)}")
    weights = getattr(torchvision.models.detection, BUILDERS[name]).COCO_V1 if pretrained else None
    model = getattr(torchvision.models.detection, name)(
        weights=weights, weights_backbone=None, min_size=imgsz, max_size=int(imgsz * 5 / 3),
        box_detections_per_img=300, box_score_thresh=score_thresh)
    cls_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(cls_in, 2)  # background + 1 class
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, 2)
    return model


class CocoInstances:
    """Torch dataset over a Roboflow-style COCO folder: image (RGB float 0..1), boxes, labels, masks."""

    def __init__(self, folder: Path, train: bool):
        d = json.loads((folder / "_annotations.coco.json").read_text(encoding="utf-8"))
        self.folder, self.train = folder, train
        self.images = [i for i in d["images"]]
        self.anns: dict[int, list] = {}
        for a in d["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        import torch

        info = self.images[i]
        bgr = cv2.imread(str(self.folder / info["file_name"]))
        h, w = bgr.shape[:2]
        masks = []
        for a in self.anns.get(info["id"], []):
            m = np.zeros((h, w), np.uint8)
            for poly in a["segmentation"]:
                pts = np.asarray(poly, np.float32).reshape(-1, 2)
                if len(pts) >= 3:
                    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
            if m.sum() >= 4:
                masks.append(m)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.train:
            if np.random.rand() < 0.5:
                img, masks = img[:, ::-1], [m[:, ::-1] for m in masks]
            if np.random.rand() < 0.5:
                img, masks = img[::-1], [m[::-1] for m in masks]
            k = np.random.randint(4)
            if k:
                img, masks = np.rot90(img, k), [np.rot90(m, k) for m in masks]
        img = np.ascontiguousarray(img)
        masks = [np.ascontiguousarray(m) for m in masks]
        boxes = []
        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        t_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.ones(len(masks), dtype=torch.int64),
                  "masks": torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, *img.shape[:2]), dtype=torch.uint8)}
        return t_img, target


def _collate(batch):  # module level: DataLoader workers on Windows must be able to pickle it
    return tuple(zip(*batch))


def coco_folder(spec: RunSpec, split: str) -> Path:
    return view_dir(spec, coco=True) / split


def _atomic_save(obj, path: Path) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class TorchvisionFamily:
    name = "torchvision"
    supports_resume = True

    def best(self, run_dir: Path) -> Path:  # no early stopping: the last epoch is the model
        return run_dir / "train" / "last.pt"

    def last(self, run_dir: Path) -> Path:
        return run_dir / "train" / "last.pt"

    def can_resume(self, run_dir: Path) -> bool:
        return self.last(run_dir).exists()

    def train(self, spec: RunSpec, run_dir: Path, resume: bool, batch: int) -> dict:
        import torch

        out = run_dir / "train"
        out.mkdir(parents=True, exist_ok=True)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.set_per_process_memory_fraction(0.92)  # fail fast instead of spilling into shared memory
        torch.manual_seed(spec.seed)
        np.random.seed(spec.seed)
        model = build_model(spec.model, spec.imgsz).to(dev)
        params = [p for p in model.parameters() if p.requires_grad]
        lr = float(spec.extra.get("lr", 0.01 * batch / 8))
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
        loader = torch.utils.data.DataLoader(
            CocoInstances(coco_folder(spec, "train"), True), batch_size=batch, shuffle=True, num_workers=2, drop_last=True,
            collate_fn=_collate, persistent_workers=True)
        total = spec.epochs * len(loader)
        warm = min(500, len(loader))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (
            0.5 * (1 + np.cos(np.pi * min(s, total) / total)) * 0.99 + 0.01))
        start, log = 0, []
        if resume and self.can_resume(run_dir):
            ck = torch.load(self.last(run_dir), map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start, log = ck["epoch"], ck["log"]
        t0 = time.time() - (log[-1]["time"] if log else 0)
        stop_at = int(os.environ.get("BENCH_TEST_INTERRUPT_EPOCH", 0))  # test hook, same as the Ultralytics family
        for epoch in range(start, spec.epochs):
            model.train()
            losses = []
            for imgs, tgts in loader:
                imgs = [i.to(dev) for i in imgs]
                tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
                loss = sum(model(imgs, tgts).values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"perda não finita na época {epoch + 1}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                sched.step()
                losses.append(float(loss))
            log.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
                        "time": time.time() - t0})
            _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                          "epoch": epoch + 1, "log": log, "name": spec.model, "imgsz": spec.imgsz}, self.last(run_dir))
            write_json(out / "log.json", log)
            if stop_at and epoch + 1 >= stop_at:
                raise KeyboardInterrupt("interrupção simulada (teste)")
        return {"vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else None}

    def model_info(self, run_dir: Path) -> dict:
        import torch

        ck = torch.load(self.best(run_dir), map_location="cpu", weights_only=False)
        n = sum(v.numel() for v in ck["model"].values() if v.dtype.is_floating_point)
        return {"params_millions": round(n / 1e6, 2), "weights_mb": round(self.best(run_dir).stat().st_size / 1e6, 1)}

    def make_predictor(self, spec: RunSpec, run_dir: Path, conf: float = 0.001):
        return TorchvisionPredictor(self.best(run_dir), conf=conf)

    def training_log(self, run_dir: Path) -> dict:
        f = run_dir / "train" / "log.json"
        if not f.exists():
            return {}
        log = json.loads(f.read_text(encoding="utf-8"))
        return {"epochs_run": len(log), "train_seconds": float(log[-1]["time"])}


class TorchvisionPredictor:
    family = "torchvision"

    def __init__(self, weights: Path, conf: float = 0.001, device: str | None = None):
        import torch

        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(ck["name"], ck["imgsz"], pretrained=False, score_thresh=0.001)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.conf = conf

    def predict(self, bgr: np.ndarray, cls_id: int | None = None) -> list[Inst]:
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.no_grad():
            r = self.model([x])[0]
        keep = (r["scores"] >= self.conf).cpu().numpy()
        masks = (r["masks"][:, 0] > 0.5).cpu().numpy()[keep]
        scores = r["scores"].cpu().numpy()[keep]
        out = []
        for m, s in zip(masks, scores):
            it = from_full(m, float(s))
            if it is not None:
                out.append(it)
        return out
