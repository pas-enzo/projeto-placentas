"""Numeros do Cap. 4 (Analise de Resultados) calculados dos dados brutos de benchmark_results.
Grava temp_ml/tcc/cap4_numeros.json e imprime conferencias com o texto antigo.
Unidades: micro f=(50/469)^2*(4140/640)*(3096/640) ; capilar f=(50/469)^2 (px -> um2).
Area 'soma' = soma das areas das instancias (definicao do Cap. 3); 'uniao' = area da uniao das mascaras.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(r"D:\projeto_placentas_clayton\benchmark_results")
OUT = Path(r"D:\projeto_placentas_clayton\temp_ml\tcc"); OUT.mkdir(parents=True, exist_ok=True)
F_MICRO = (50 / 469) ** 2 * (4140 / 640) * (3096 / 640)
F_CAP = (50 / 469) ** 2
MARES = ["ROSILHA-M", "TORDILHA", "TP", "ZAINA-679", "ZAZA", "TOSTADA"]
d = pd.read_csv(R / "results.csv")
d = d[d.norm == "none"].copy()
d["mare"] = d.fold.map(lambda f: "ZAZA" if f == "primary" else f.split("_val-")[0].replace("test-", ""))


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 2 * np.mean((x - x.mean()) * (y - y.mean())) / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def fields(task, model):
    rows = []
    for _, r in d[(d.task == task) & (d.model == model)].iterrows():
        p = json.load(open(R / r.run_id / "perimage_test.json"))
        rr = p["image"]["rows"] if task == "micro" else p["field"]["rows"]
        f = F_MICRO if task == "micro" else F_CAP
        for i, x in enumerate(rr):
            rows.append(dict(mare=r.mare, i=i, tp=x["tp"], fp=x["fp"], fn=x["fn"], n_gt=x["n_gt"], n_pred=x["n_pred"],
                             gt_sum=x["area_gt_sum"] * f, pr_sum=x["area_pred_sum"] * f, gt_uni=x["area_gt_union"] * f,
                             pr_uni=x["area_pred_union"] * f, inter=x["inter"]))
    return pd.DataFrame(rows)


def summary(F):
    o = {}
    for a in ("sum", "uni"):
        g, p = F[f"gt_{a}"], F[f"pr_{a}"]
        da = p - g
        o[a] = dict(gt_mean=g.mean(), pr_mean=p.mean(), r=np.corrcoef(g, p)[0, 1], ccc=ccc(g, p), bias=da.mean(),
                    loa=[da.mean() - 1.96 * da.std(), da.mean() + 1.96 * da.std()],
                    err_mean=(da.abs() / g).mean() * 100, err_med=(da.abs() / g).median() * 100, bias_pct=(da / g).mean() * 100)
    dn = F.n_pred - F.n_gt
    o["n"] = dict(gt_mean=F.n_gt.mean(), pr_mean=F.n_pred.mean(), r=np.corrcoef(F.n_gt, F.n_pred)[0, 1], ccc=ccc(F.n_gt, F.n_pred),
                  bias=dn.mean(), loa=[dn.mean() - 1.96 * dn.std(), dn.mean() + 1.96 * dn.std()], err_mean=dn.abs().mean(), err_med=dn.abs().median())
    return o


res = {"fields": {}, "mare": {}, "six": {}}
for task, models in (("micro", ["yolo26s-seg.pt", "yolo26m-seg.pt", "yolo11m-seg.pt"]), ("capilar", ["yolo11s-seg.pt", "yolo26s-seg.pt"])):
    for m in models:
        F = fields(task, m)
        F.to_csv(OUT / f"campos_{task}_{m.split('-')[0]}.csv", index=False)
        res["fields"][f"{task}/{m}"] = summary(F)
        pre = "test_report" if task == "micro" else "test_field"
        s = d[(d.task == task) & (d.model == m)].set_index("mare").loc[MARES]
        res["six"][f"{task}/{m}"] = {k: [float(s[c].mean()), float(s[c].std())] for k, c in
            dict(f1=pre + "_instance_f1", prec=pre + "_instance_precision", rec=pre + "_instance_recall", dice=pre + "_pixel_dice_pooled",
                 val_ap=None).items() if c}
        res["six"][f"{task}/{m}"]["val_ap"] = [float(s.val_report_coco_ap_ap.mean()), float(s.val_report_coco_ap_ap.std())]
        if task == "micro":
            res["six"][f"{task}/{m}"]["test_ap"] = [float(s.test_report_coco_ap_ap.mean()), float(s.test_report_coco_ap_ap.std())]
        res["six"][f"{task}/{m}"]["params_M"] = float(s.model_params_millions.iloc[0])
        res["six"][f"{task}/{m}"]["val_ap_per_mare"] = {k: float(v) for k, v in s.val_report_coco_ap_ap.items()}
        res["six"][f"{task}/{m}"]["conf_test"] = {k: float(v) for k, v in s[pre + "_conf" if task == "micro" else "test_conf_star"].items()}
        res["mare"][f"{task}/{m}"] = {mm: dict(f1=float(s.loc[mm, pre + "_instance_f1"]), prec=float(s.loc[mm, pre + "_instance_precision"]),
                                              rec=float(s.loc[mm, pre + "_instance_recall"]), dice=float(s.loc[mm, pre + "_pixel_dice_pooled"]),
                                              area_err_sum=float(s.loc[mm, pre + "_morphometry_area_sum_mean_abs_rel_error"] * 100),
                                              area_err_uni=float(s.loc[mm, pre + "_morphometry_area_union_mean_abs_rel_error"] * 100),
                                              count_err=float(s.loc[mm, pre + "_morphometry_count_mean_abs_error"])) for mm in MARES}
json.dump(res, open(OUT / "cap4_numeros.json", "w"), indent=1, default=float)

# ---- conferencias com o texto antigo
print("MICRO 26s seis egua: ", {k: [round(x, 3) for x in v] for k, v in res["six"]["micro/yolo26s-seg.pt"].items() if isinstance(v, list)})
print("CAP 11s seis egua: ", {k: [round(x, 3) for x in v] for k, v in res["six"]["capilar/yolo11s-seg.pt"].items() if isinstance(v, list)})
for k in ("micro/yolo26s-seg.pt", "capilar/yolo11s-seg.pt"):
    s = res["fields"][k]
    print(k, "SOMA", {a: round(v, 3) if not isinstance(v, list) else [round(x, 1) for x in v] for a, v in s["sum"].items()})
    print(k, "UNIAO", {a: round(v, 3) if not isinstance(v, list) else [round(x, 1) for x in v] for a, v in s["uni"].items()})
    print(k, "N", {a: round(v, 3) if not isinstance(v, list) else [round(x, 1) for x in v] for a, v in s["n"].items()})
