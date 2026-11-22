"""Tabelas completas do Cap. 4 (Analise de Resultados), geradas dos dados brutos, para o Apendice B da monografia.

Saida: D:/Downloads/projeto_tcc_enzo_template_correto/graduacao/chapters/apendice_resultados.tex
       temp_ml/tcc/cap4_pareado_finalistas.csv (comparacao pareada dos finalistas, 180 campos)
Todos os numeros vem de benchmark_results (results.csv e perimage_test.json). Area 'soma' = soma das areas das instancias.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
import figuras_cap4 as F                                     # noqa: E402  (dados e estatistica de normalizacao)
from benchmark.report_stats import stat_values, holm        # noqa: E402

R = F.R
D = F.D
OUT = Path(r"D:\Downloads\projeto_tcc_enzo_template_correto\graduacao\chapters\apendice_resultados.tex")
MARES = F.MARES
NOME = F.NOME


def c(x, n=2):                       # virgula decimal
    return f"{x:.{n}f}".replace(".", ",")


def sgn(x, n=3):
    return ("$+$" if x >= 0 else "$-$") + c(abs(x), n)


def milhar(x):
    return f"{x:,.0f}".replace(",", ".")


def prim(task, model, norm="none", epochs=None):
    s = D[(D.task == task) & (D.model == model) & (D.norm == norm) & (D.fold == "primary")]
    if epochs is not None:
        s = s[s.train_epochs_run == epochs]
    return s.iloc[0]


def row_variante(task, r, nome):
    pre = "test_report" if task == "micro" else "test_field"
    conf = r.test_report_conf if task == "micro" else r.val_conf_star
    ap = r.test_report_coco_ap_ap
    return (f"{nome} & {c(r.model_params_millions, 1)} & {c(conf)} & {c(ap, 3)} & {c(r[pre + '_instance_precision'], 3)} & "
            f"{c(r[pre + '_instance_recall'], 3)} & {c(r[pre + '_instance_f1'], 3)} & {c(r[pre + '_pixel_dice_pooled'], 3)} & "
            f"{c(r[pre + '_morphometry_area_sum_mean_abs_rel_error'] * 100, 1)}\\% & {c(r[pre + '_morphometry_count_mean_abs_error'], 1)} \\\\")


def tabela_variantes():
    L = []
    for task, title, label, models, extra in (
        ("micro", "Comparação das variantes de microcotilédones no fold principal (teste: égua ZAZA, 30 campos).", "tab:comp-micro",
         ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov9c-seg.pt", "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
          "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt"],
         ["maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2", "rfdetr-seg-nano"]),
        ("capilar", "Comparação das variantes de capilares no fold principal (teste: égua ZAZA, 30 campos inteiros).", "tab:comp-cap",
         ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov9c-seg.pt", "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
          "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt"],
         ["maskrcnn_resnet50_fpn"])):
        L += ["\\begin{table}[htb]", "    \\centering", f"    \\caption{{{title}}}", f"    \\label{{{label}}}", "    \\footnotesize\n    \\setlength{\\tabcolsep}{3.5pt}",
              "    \\begin{tabular}{lcccccccccc}".replace("cccccccccc", "ccccccccc"), "        \\hline",
              "        Modelo & Param. (M) & Conf.* & AP & Prec. & Rev. & F1 & Dice & Erro de área & Erro de contagem \\\\", "        \\hline"]
        for m in models:
            nome = NOME[m] + (" $^{\\dagger}$" if task == "capilar" and m in {"yolov8m-seg.pt", "yolov9c-seg.pt", "yolo11m-seg.pt", "yolo26m-seg.pt"} else "")
            L.append("        " + row_variante(task, prim(task, m), nome))
        L.append("        \\hline")
        for m in extra:
            if m == "maskrcnn_resnet50_fpn" and task == "capilar":
                for ep in (20, 40):
                    L.append("        " + row_variante(task, prim(task, m, epochs=ep), f"Mask R-CNN v1 ({ep} ép.) $^{{\\dagger}}$"))
            else:
                L.append("        " + row_variante(task, prim(task, m), NOME[m] + " $^{\\dagger}$"))
        nota = ("Conf.*: limiar de confiança calibrado na validação. Prec.: precisão; Rev.: revocação. O AP é o AP de máscara do COCO"
                + ("; nos capilares, é calculado por ladrilho, e as demais métricas, no campo inteiro" if task == "capilar" else "")
                + ". Erro de área: erro relativo absoluto médio da soma das áreas por campo; erro de contagem: erro absoluto médio por campo. "
                "$^{\\dagger}$Treinado depois da definição dos finalistas (Seção~\\ref{sec:selecao}). A linha abaixo da divisória reúne os comparativos de outras famílias.")
        L += ["        \\hline", "    \\end{tabular}", f"    \\\\ {{\\footnotesize \\textbf{{Fonte:}} Elaborada pelo autor (2026). {nota}}}", "\\end{table}", ""]
    return L


def pm(task, m):
    pre = "test_report" if task == "micro" else "test_field"
    s = F.six(task, m)
    return s, pre


def tabela_egua(task, m, titulo, label):
    s, pre = pm(task, m)
    L = ["\\begin{table}[htb]", "    \\centering", f"    \\caption{{{titulo}}}", f"    \\label{{{label}}}", "    \\footnotesize\n    \\setlength{\\tabcolsep}{3.5pt}",
         "    \\begin{tabular}{lcccccccc}", "        \\hline",
         "        Égua de teste & Conf.* & " + ("AP & " if task == "micro" else "") + "Prec. & Rev. & F1 & Dice & Erro de área & Erro de contagem \\\\",
         "        \\hline"]
    L[5] = "    \\begin{tabular}{l" + "c" * (8 if task == "micro" else 7) + "}"
    vals = {k: [] for k in ("ap", "p", "r", "f1", "d", "ea", "ce")}
    for mm in MARES:
        r = s.loc[mm]
        conf = r.test_report_conf if task == "micro" else r.val_conf_star
        v = dict(ap=r.test_report_coco_ap_ap, p=r[pre + "_instance_precision"], r=r[pre + "_instance_recall"], f1=r[pre + "_instance_f1"],
                 d=r[pre + "_pixel_dice_pooled"], ea=r[pre + "_morphometry_area_sum_mean_abs_rel_error"] * 100, ce=r[pre + "_morphometry_count_mean_abs_error"])
        for k in vals:
            vals[k].append(v[k])
        L.append(f"        {mm} & {c(conf)} & " + (f"{c(v['ap'], 3)} & " if task == "micro" else "") +
                 f"{c(v['p'], 3)} & {c(v['r'], 3)} & {c(v['f1'], 3)} & {c(v['d'], 3)} & {c(v['ea'], 1)}\\% & {c(v['ce'], 1)} \\\\")
    L.append("        \\hline")
    ms = lambda k, n: f"${c(np.mean(vals[k]), n)} \\pm {c(np.std(vals[k], ddof=1), n)}$"   # noqa: E731
    L.append("        Média $\\pm$ dp & -- & " + (f"{ms('ap', 3)} & " if task == "micro" else "") +
             f"{ms('p', 3)} & {ms('r', 3)} & {ms('f1', 3)} & {ms('d', 3)} & {ms('ea', 1)} & {ms('ce', 1)} \\\\")
    L += ["        \\hline", "    \\end{tabular}",
          "    \\\\ {\\footnotesize \\textbf{Fonte:} Elaborada pelo autor (2026). Cada linha é uma rotação em que aquela égua é o teste; Conf.*: limiar calibrado na validação "
          "da rotação. dp: desvio-padrão amostral entre as seis éguas. Erro de área: da soma das áreas das instâncias.}", "\\end{table}", ""]
    return L


def pooled_rows(task, model):
    s = F.six(task, model)
    out = []
    for mm in MARES:
        p = json.load(open(R / s.loc[mm, "run_id"] / "perimage_test.json"))
        rr = p["image"]["rows"] if task == "micro" else p["field"]["rows"]
        out += [dict(mare=mm, **x) for x in rr]
    return pd.DataFrame(out)


def pareado():
    """26s x 11m e 26s x 26m (micro); 11s x 26s (capilar): 180 campos reunidos, bootstrap pareado de 2000 reamostragens, Holm por tarefa."""
    idx = np.random.default_rng(0).integers(0, 180, (2000, 180))
    res = []
    for task, ref, others in (("micro", "yolo26s-seg.pt", ["yolo11m-seg.pt", "yolo26m-seg.pt"]), ("capilar", "yolo11s-seg.pt", ["yolo26s-seg.pt"])):
        pr = pooled_rows(task, ref)
        bt = pd.DataFrame([stat_values(pr, i) for i in idx])
        sub = []
        for o in others:
            po = pooled_rows(task, o)
            assert (pr.n_gt.values == po.n_gt.values).all(), "campos fora de ordem entre modelos"
            bo = pd.DataFrame([stat_values(po, i) for i in idx])
            for st in ("f1", "dice", "area_err", "count_mae"):
                dd = bt[st].to_numpy() - bo[st].to_numpy()
                p = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / 2000)
                sub.append(dict(task=task, ref=ref, other=o, stat=st, diff=stat_values(pr)[st] - stat_values(po)[st],
                                lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5), p=p))
        t = pd.DataFrame(sub)
        t["p_holm"] = holm(t.p.to_numpy())
        res.append(t)
    return pd.concat(res, ignore_index=True)


def tabela_pareado(t):
    nm = {"f1": "F1", "dice": "Dice", "area_err": "Erro de área", "count_mae": "Erro de contagem"}
    L = ["\\begin{table}[htb]", "    \\centering",
         "    \\caption{Comparação pareada entre os finalistas, com os 180 campos de teste das seis rotações.}", "    \\label{tab:paired}", "    \\footnotesize\n    \\setlength{\\tabcolsep}{3.5pt}",
         "    \\begin{tabular}{llccrr}", "        \\hline", "        Comparação & Estatística & Diferença & IC 95\\% & $p$ & $p$ (Holm) \\\\", "        \\hline"]
    first = True
    for (task, o), g in t.groupby(["task", "other"], sort=False):
        if not first:
            L.append("        \\hline")
        first = False
        for _, r in g.iterrows():
            n = 2 if r.stat == "count_mae" else 3
            L.append(f"        {NOME[r.ref]} $-$ {NOME[r.other]} & {nm[r.stat]} & {sgn(r['diff'], n)} & [{sgn(r.lo, n)}; {sgn(r.hi, n)}] & {c(r.p, 3)} & {c(r.p_holm, 3)} \\\\")
    L += ["        \\hline", "    \\end{tabular}",
          "    \\\\ {\\footnotesize \\textbf{Fonte:} Elaborada pelo autor (2026). Diferença do primeiro modelo menos o segundo; para os erros, valores negativos favorecem o primeiro. "
          "\\textit{Bootstrap} de 2.000 reamostragens dos campos; Holm sobre as comparações de cada tarefa (oito nos microcotilédones, quatro nos capilares). "
          "Erro de área: erro relativo da soma das áreas, em fração.}", "\\end{table}", ""]
    return L


def tabela_norm():
    st = F.stats_norm()
    st = st[st.stat == "f1"]
    order = ["none", "whitebalance", "grayscale", "histmatch", "reinhard", "macenko", "vahadane"]
    cols = ["Sem norm.", "Balanço de branco", "Escala de cinza", "\\textit{Histmatch}", "Reinhard", "Macenko", "Vahadane"]
    L = ["\\begin{table}[htb]", "    \\centering",
         "    \\caption{F1 e AP com cada método de normalização de coloração, no fold principal (teste: égua ZAZA, 30 campos).}", "    \\label{tab:norm-abs}", "    \\footnotesize\n    \\setlength{\\tabcolsep}{3.5pt}",
         "    \\begin{tabular}{llccccccc}", "        \\hline", "        Tarefa e modelo & Medida & " + " & ".join(cols) + " \\\\", "        \\hline"]
    prim_ = D[D.fold == "primary"]
    spec = [("micro", ["yolo26s-seg.pt", "yolo26m-seg.pt", "yolo11m-seg.pt"]), ("capilar", ["yolo11n-seg.pt", "yolo11s-seg.pt"])]
    first = True
    for task, mods in spec:
        if not first:
            L.append("        \\hline")
        first = False
        pre = "test_report" if task == "micro" else "test_field"
        for m in mods:
            rows = {n: prim_[(prim_.task == task) & (prim_.model == m) & (prim_.norm == n)] for n in order}
            f1 = [float(rows[n][pre + "_instance_f1"].iloc[0]) for n in order]
            cel = []
            for n, v in zip(order, f1):
                if n == "none":
                    cel.append(c(v, 3))
                    continue
                r = st[(st.task == task) & (st.model == m) & (st.norm == n)]
                mark = ""
                if len(r) and r.p_holm.iloc[0] < .05:
                    mark = "$^{*}$"
                if not len(r):
                    mark = "$^{\\dagger}$"
                cel.append(c(v, 3) + mark)
            L.append(f"        {'Microcotilédones' if task == 'micro' else 'Capilares'}, {NOME[m]} & F1 & " + " & ".join(cel) + " \\\\")
            if task == "micro":
                ap = [float(rows[n]["test_report_coco_ap_ap"].iloc[0]) for n in order]
                L.append("         & AP & " + " & ".join(c(v, 3) for v in ap) + " \\\\")
    L += ["        \\hline", "    \\end{tabular}",
          "    \\\\ {\\footnotesize \\textbf{Fonte:} Elaborada pelo autor (2026). Uma semente por configuração; a referência de cor foi estimada só com as éguas de treino. "
          "$^{*}$Diferença de F1 em relação ao mesmo modelo sem normalização com $p$ ajustado de Holm menor que 0,05 (\\textit{bootstrap} pareado dos 30 campos, 2.000 reamostragens). "
          "$^{\\dagger}$Sem teste pareado: não há estatísticas por campo para essas execuções, porque a estimativa da matriz de cores do método de Vahadane não é reprodutível. "
          "O AP dos capilares, calculado por ladrilho, não é mostrado.}", "\\end{table}", ""]
    return L


def tabela_conc():
    L = []
    for task, mod, titulo, label in (("micro", "yolo26s-seg.pt", "Concordância entre o YOLO26s e a anotação, campo a campo, para os microcotilédones.", "tab:conc-micro"),
                                     ("capilar", "yolo11s-seg.pt", "Concordância entre o YOLO11s e a anotação, campo a campo, para os capilares.", "tab:conc-cap")):
        s = F.six(task, mod)
        pre = "test_report" if task == "micro" else "test_field"
        zaza = s.loc["ZAZA"]
        C = F.campos(task, mod)
        full = pd.DataFrame([dict(mare=mm, **x) for mm in MARES for x in
                             (json.load(open(R / s.loc[mm, "run_id"] / "perimage_test.json"))["image" if task == "micro" else "field"]["rows"])])
        f = F.F_MICRO if task == "micro" else F.F_CAP

        def stats(sub, kind):
            if kind == "n":
                g, p = sub.n_gt.values.astype(float), sub.n_pred.values.astype(float)
            elif kind == "sum":
                g, p = sub.area_gt_sum.values * f, sub.area_pred_sum.values * f
            else:
                g, p = sub.area_gt_union.values * f, sub.area_pred_union.values * f
            dd = p - g
            return np.corrcoef(g, p)[0, 1], F.ccc(g, p), dd.mean(), dd.mean() - 1.96 * dd.std(ddof=1), dd.mean() + 1.96 * dd.std(ddof=1)

        L += ["\\begin{table}[htb]", "    \\centering", f"    \\caption{{{titulo}}}", f"    \\label{{{label}}}", "    \\footnotesize\n    \\setlength{\\tabcolsep}{3.5pt}",
              "    \\begin{tabular}{llcccc}", "        \\hline", "        Conjunto de teste & Medida & Pearson & CCC & Viés & Limites de concordância (95\\%) \\\\", "        \\hline"]
        for nome, sub in (("ZAZA (30 campos)", full[full.mare == "ZAZA"]), ("Seis éguas (180 campos)", full)):
            for kind, lab in (("n", "Contagem"), ("sum", "Área total (soma, $\\mu$m$^2$)"), ("uni", "Área da união ($\\mu$m$^2$)")):
                r_, cc_, b, lo, hi = stats(sub, kind)
                fm = (lambda v: c(v, 1)) if kind == "n" else (lambda v: milhar(v))   # noqa: E731
                L.append(f"        {nome} & {lab} & {c(r_)} & {c(cc_)} & {fm(b).replace('-', '$-$')} & [{fm(lo).replace('-', '$-$')}; {fm(hi).replace('-', '$-$')}] \\\\")
            if nome.startswith("ZAZA"):
                L.append("        \\hline")
        L += ["        \\hline", "    \\end{tabular}",
              "    \\\\ {\\footnotesize \\textbf{Fonte:} Elaborada pelo autor (2026). Viés: média de (predito $-$ anotado). CCC: coeficiente de correlação de concordância de Lin. "
              "As éguas foram avaliadas cada uma na sua rotação.}", "\\end{table}", ""]
    return L


def tabela_tempos():
    j = json.load(open(R / "inference_timing.json"))["rows"]
    L = ["\\begin{table}[htb]", "    \\centering", "    \\caption{Tempo de inferência dos finalistas, por imagem (microcotilédones) e por campo nativo (capilares).}", "    \\label{tab:time}", "    \\footnotesize\n    \\setlength{\\tabcolsep}{3.5pt}",
         "    \\begin{tabular}{llcccccc}", "        \\hline", "        Tarefa & Modelo & FP32 (média) & dp & P95 & FP16 (média) & VRAM (MB) & Conf. \\\\",
         "        \\hline"]
    for r in j:
        un, sc, nd = ("ms", 1e3, 0) if r["task"] == "micro" else ("s", 1, 1)
        L.append(f"        {'Microcotilédones' if r['task'] == 'micro' else 'Capilares'} & {NOME[r['model']]} & {c(r['fp32']['mean_s'] * sc, nd)} {un} & "
                 f"{c(r['fp32']['sd_s'] * sc, nd)} {un} & {c(r['fp32']['p95_s'] * sc, nd)} {un} & {c(r['fp16']['mean_s'] * sc, nd)} {un} & "
                 f"{r['fp32']['vram_peak_mb']:.0f} & {c(r['conf'])} \\\\")
    L += ["        \\hline", "    \\end{tabular}",
          "    \\\\ {\\footnotesize \\textbf{Fonte:} Elaborada pelo autor (2026). GTX 1650 SUPER, uma imagem por vez, 30 imagens ou campos de teste, aquecimento descartado; "
          "tempo de ponta a ponta com a confiança calibrada. VRAM: pico em FP32. dp: desvio-padrão amostral; P95: percentil 95.}", "\\end{table}", ""]
    return L


if __name__ == "__main__":
    t = pareado()
    t.to_csv(r"D:\projeto_placentas_clayton\temp_ml\tcc\cap4_pareado_finalistas.csv", index=False)
    print(t.round(4).to_string())
    head = ["\\chapter{Resultados complementares}", "\\label{ap:resultados}", "",
            "Este apêndice reúne as tabelas completas que sustentam o Capítulo~\\ref{c.analise_resultados}: o desempenho de todas as variantes no fold principal (Tabelas~\\ref{tab:comp-micro} e \\ref{tab:comp-cap}), "
            "o desempenho por égua dos modelos adotados (Tabelas~\\ref{tab:mare-micro} e \\ref{tab:mare-cap}), a comparação pareada dos finalistas (Tabela~\\ref{tab:paired}), "
            "os resultados absolutos com cada normalização de coloração (Tabela~\\ref{tab:norm-abs}), a concordância das medidas com a anotação (Tabelas~\\ref{tab:conc-micro} e \\ref{tab:conc-cap}) "
            "e os tempos de inferência (Tabela~\\ref{tab:time}).", ""]
    body = (tabela_variantes() + tabela_egua("micro", "yolo26s-seg.pt", "Desempenho do YOLO26s nos microcotilédones em cada égua de teste.", "tab:mare-micro")
            + tabela_egua("capilar", "yolo11s-seg.pt", "Desempenho do YOLO11s nos capilares (campo inteiro) em cada égua de teste.", "tab:mare-cap")
            + tabela_pareado(t) + tabela_norm() + tabela_conc() + tabela_tempos())
    OUT.write_text("\n".join(head + body), encoding="utf-8")
    print("apendice:", OUT, len(body), "linhas")
