"""One adapter per model family.

A family object exposes: `name`, `supports_resume`, `train(spec, run_dir, resume, batch)`, `best(run_dir)`,
`can_resume(run_dir)`, `model_info(run_dir)`, `make_predictor(spec, run_dir, conf)` and `training_log(run_dir)`;
the predictor exposes `predict(bgr, cls_id=None) -> list[Inst]` and a mutable `conf`.
"""


def get_family(name: str):
    if name == "ultralytics":
        from .ultralytics_train import UltralyticsFamily

        return UltralyticsFamily()
    raise KeyError(f"família sem adaptador ainda: {name!r}")
