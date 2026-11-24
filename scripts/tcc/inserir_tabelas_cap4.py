"""Insere no corpo do Cap. 4 (analise_resultados_reescrito.tex) as tabelas geradas por tabelas_cap4.py, nos pontos onde sao citadas.
Idempotente: as tabelas inseridas ficam entre marcadores '% >>> tab:xxx' e '% <<< tab:xxx' e sao refeitas a cada execucao.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tabelas_cap4 as T   # noqa: E402

CAP = Path(r"D:\Downloads\projeto_tcc_enzo_template_correto\graduacao\chapters\analise_resultados_reescrito.tex")
NL = "\n"


def tabelas():
    t = T.pareado()
    linhas = (T.tabela_variantes() + T.tabela_egua("micro", "yolo26s-seg.pt", "Desempenho do YOLO26s nos microcotilédones em cada égua de teste.", "tab:mare-micro")
              + T.tabela_egua("capilar", "yolo11s-seg.pt", "Desempenho do YOLO11s nos capilares (campo inteiro) em cada égua de teste.", "tab:mare-cap")
              + T.tabela_pareado(t) + T.tabela_norm() + T.tabela_conc() + T.tabela_tempos())
    txt = NL.join(linhas)
    blocos = re.split(r"(?<=\\end\{table\})\s*\n", txt)
    out = {}
    for b in blocos:
        b = b.strip()
        if not b:
            continue
        m = re.search(r"\\label\{(tab:[^}]+)\}", b)
        # nunca passa da largura do texto: reduz a tabela se for mais larga, e nao amplia se for mais estreita
        b = b.replace("    \\begin{tabular}", "    \\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n    \\begin{tabular}", 1)
        b = b.replace("    \\end{tabular}", "    \\end{tabular}}", 1)
        out[m.group(1)] = b
    return out


# tabela -> trecho do capitulo ANTES do qual ela entra
ANTES = [
    (["tab:comp-cap"], r"\subsection{Finalistas nas seis rotações}"),
    (["tab:paired"], r"\section{Desempenho dos modelos adotados}"),
    (["tab:mare-micro", "tab:mare-cap"], r"\subsection{O caso da égua TOSTADA}"),
    (["tab:norm-abs"], r"\section{Concordância das medidas com a anotação}"),
    (["tab:conc-micro"], r"\subsection{Capilares}"),
    (["tab:conc-cap"], r"\section{Tempo de inferência e ganho de eficiência}"),
    (["tab:time"], r"\subsection{Anotação manual e \textit{pipeline}}"),
]
# tabela -> trecho do capitulo DEPOIS do qual (fim do paragrafo) ela entra
DEPOIS = [(["tab:comp-micro"], "foi treinada com outra receita (Seção~\\ref{sec:modelos-outros}).")]


def limpar(s):
    return re.sub(r"(?s)% >>> tab:[^\n]*\n.*?% <<< tab:[^\n]*\n\n?", "", s)


def main():
    s = limpar(CAP.read_text(encoding="utf-8"))
    tab = tabelas()
    s = s.replace("As tabelas completas estão no Apêndice~\\ref{ap:resultados}, e o corpo do capítulo traz os resultados em figuras.",
                  "As figuras mostram os resultados, e as tabelas trazem os valores correspondentes.")
    for nomes, ancora in ANTES:
        assert ancora in s, ancora
        bloco = NL.join(f"% >>> {n}{NL}{tab[n]}{NL}% <<< {n}{NL}" for n in nomes) + NL
        s = s.replace(ancora, bloco + ancora, 1)
    for nomes, frase in DEPOIS:
        assert frase in s, frase
        i = s.index(frase) + len(frase)
        j = s.index(NL, i)
        bloco = NL + NL + NL.join(f"% >>> {n}{NL}{tab[n]}{NL}% <<< {n}{NL}" for n in nomes)
        s = s[:j] + bloco + s[j:]
    CAP.write_text(s, encoding="utf-8", newline=NL)
    usados = re.findall(r"% >>> (tab:[^\n]+)", s)
    print("inseridas:", usados)
    assert set(usados) == set(tab), set(tab) ^ set(usados)


if __name__ == "__main__":
    main()
