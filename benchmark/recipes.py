"""Training recipes (augmentation / loss settings) per task, taken from the champion runs of the project.

Fairness rule: inside one family, every model of a task uses the SAME recipe; only the model changes.
`default` leaves the framework defaults untouched.
"""

RECIPES = {
    "default": {},
    # YOLO micro champion (placentas_v11_aug_v2): strong augmentation, full-resolution masks
    "micro_champion": {
        "mosaic": 1.0, "mixup": 0.1, "degrees": 90.0, "translate": 0.1, "scale": 0.5, "flipud": 0.5, "fliplr": 0.5,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "close_mosaic": 10, "erasing": 0.4,
        "auto_augment": "randaugment", "overlap_mask": False, "mask_ratio": 1,
    },
    # Same augmentation as the micro champion but with the library-default mask resolution (mask_ratio 4): the champion's
    # mask_ratio 1 does not fit a 4 GB GPU with current Ultralytics. Benchmark rule: ONE recipe per task for all models.
    "micro_bench": {
        "mosaic": 1.0, "mixup": 0.1, "degrees": 90.0, "translate": 0.1, "scale": 0.5, "flipud": 0.5, "fliplr": 0.5,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "close_mosaic": 10, "erasing": 0.4,
        "auto_augment": "randaugment", "overlap_mask": False, "mask_ratio": 4,
    },
    # YOLO capillary champion (capilar_yolo11s_tiled_v4): light mosaic, copy-paste, overlapping masks
    "capilar_champion": {
        "mosaic": 0.15, "copy_paste": 0.15, "degrees": 90.0, "translate": 0.1, "scale": 0.3, "flipud": 0.5,
        "fliplr": 0.5, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "close_mosaic": 10, "erasing": 0.4,
        "auto_augment": "randaugment", "overlap_mask": True, "mask_ratio": 4, "max_det": 150,
    },
}


def get(name: str) -> dict:
    if name not in RECIPES:
        raise KeyError(f"receita desconhecida {name!r}; opções: {list(RECIPES)}")
    return dict(RECIPES[name])
