"""Organiza o .bib do artigo por secao e separa as referencias nao usadas.

Entradas: introduction_en.tex, methods_0*.tex (chaves citadas, na ordem de primeira aparicao) e as fontes de entradas
(refs_introducao.bib, refs_metodos.bib, _fonte_entradas_overleaf.bib).
Saidas (em artigo/): referencias.bib (Introducao, depois Materiais e Metodos), e verificacao no terminal.
refs_removidas.bib foi gerado a parte (entradas do Overleaf nao citadas).
"""
import re
import sys
from pathlib import Path

ART = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\artigo")
SOURCES = ["refs_introducao.bib", "refs_metodos.bib", "_fonte_entradas_overleaf.bib"]
INTRO = ["introduction_en.tex"]
METHODS = sorted(p.name for p in ART.glob("methods_0*.tex"))
ABSTRACT = ["abstract.tex"]


def cites(files):
    keys = []
    for f in files:
        txt = re.sub(r"(?m)^\s*%.*$", "", (ART / f).read_text(encoding="utf-8"))   # ignora linhas comentadas
        for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", txt):
            for k in m.group(1).split(","):
                k = k.strip()
                if k and k not in keys:
                    keys.append(k)
    return keys


def parse(path):
    """Devolve {chave: texto da entrada} e a lista de linhas de comentario (%) fora das entradas."""
    txt = path.read_text(encoding="utf-8")
    entries, comments, i = {}, [], 0
    while i < len(txt):
        m = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,").search(txt, i)
        if not m:
            break
        comments += [l.strip() for l in txt[i:m.start()].splitlines() if l.strip().startswith("%")]
        depth, j = 0, txt.index("{", m.start())
        while j < len(txt):
            depth += txt[j] == "{"
            depth -= txt[j] == "}"
            j += 1
            if depth == 0:
                break
        assert depth == 0, f"chaves desbalanceadas em {m.group(2)}"
        entries[m.group(2)] = txt[m.start():j].strip()
        i = j
    comments += [l.strip() for l in txt[i:].splitlines() if l.strip().startswith("%")]
    return entries, comments


entries, notes = {}, []
for s in SOURCES:
    e, c = parse(ART / s)
    dup = set(e) & set(entries)
    assert not dup, f"chave duplicada entre fontes: {dup}"
    entries.update(e)
    if s == "refs_metodos.bib":
        notes = [x for x in c if x not in notes and not x.startswith("% Referencias novas")]

intro = cites(INTRO)
meth = [k for k in cites(METHODS) if k not in intro]
abstract_keys = cites(ABSTRACT)
used = intro + meth
missing = [k for k in used + abstract_keys if k not in entries]
print(f"Introducao: {len(intro)} chaves | Materiais e Metodos (exclusivas): {len(meth)} | Resumo: {len(abstract_keys)}")
print("citadas sem entrada:", missing or "nenhuma")
print("entradas nao citadas (nas fontes usadas):", sorted(set(entries) - set(used)) or "nenhuma")
if missing:
    sys.exit(1)

KEEP_NOTE = re.compile(r"^(Accessed|Version)")   # 'note' que e informacao real da referencia (data de acesso, versao do software)


def notes_to_comments(key, entry):
    """Troca o campo note (lembrete interno) por linhas de comentario (%) acima da entrada; mantem note de acesso/versao."""
    m = re.search(r",?\s*note\s*=\s*\{", entry)
    if not m:
        return entry
    depth, j = 1, m.end()
    while depth:
        depth += entry[j] == "{"
        depth -= entry[j] == "}"
        j += 1
    text = " ".join(entry[m.end():j - 1].split())
    if KEEP_NOTE.match(text):
        return entry
    body = (entry[:m.start()] + entry[j:]).rstrip()
    body = re.sub(r",\s*\}$", "\n}", body)                       # tira a virgula que sobrou antes do fecha-chaves
    body = re.sub(r"\n\s*\n\}$", "\n}", body)
    return f"% NOTA ({key}): {text}\n{body}"


for k in list(entries):
    entries[k] = notes_to_comments(k, entries[k])

out = ["% Referencias do artigo, organizadas por secao (gerado por scripts/artigo_iser/organizar_bib.py, 2026-11-15).",
       "% Dentro de cada secao, na ordem da primeira citacao no texto; uma referencia citada em mais de uma secao fica na primeira.",
       "% Referencias retiradas (nao citadas na Introducao nem em Materiais e Metodos): refs_removidas.bib.",
       "% Os lembretes de cada referencia estao como comentario (%) acima da entrada ('NOTA'); o registro robles_xxxx e provisorio.", ""]
out += ["% ==========================================================================", "% INTRODUCTION", "% ==========================================================================", ""]
out += [entries[k] + "\n" for k in intro]
out += ["% ==========================================================================", "% MATERIALS AND METHODS", "% ==========================================================================", ""]
METHODS_NOTES = {   # notas de verificacao de cada referencia de Metodos (antes eram um bloco unico no topo do arquivo)
    "chen2021": "titulo, autores, revista, ano, volume, numero, paginas e DOI conferidos por busca (duas fontes concordam); o texto completo NAO foi lido (o site da MDPI bloqueou o acesso). Ler o artigo antes de submeter, para conferir a atribuicao da regra do um erro-padrao.",
    "he2017maskrcnn": "autores, titulo, ICCV 2017 e DOI conferidos por busca. Paginas 2961--2969 sao as do repositorio aberto da CVF; o indice do DOI mostra 2980--2988 em alguns sites. Conferir no IEEE Xplore antes de submeter.",
    "wang2024yolov9": "autores, titulo e ECCV 2024 conferidos por busca; o DOI vem de uma fonte secundaria (mlanthology) e as paginas nao foram conferidas (omitidas). Conferir na pagina da ECVA/Springer antes de submeter.",
    "reinhard2001": "titulo, autores, venue, ano, paginas e DOI conferidos por busca (registros secundarios concordam; paginas e DOI vem do dblp e de registros secundarios). Conferir em doi.org.",
    "macenko2009": "titulo, autores, venue, ano, paginas e DOI conferidos por busca (registros secundarios concordam). Conferir em doi.org.",
    "vahadane2016": "titulo, autores, venue, ano, paginas e DOI conferidos por busca (registros secundarios concordam); o DOI (TMI.2016.2529665) e o que as fontes indicam. Conferir em doi.org.",
    "tellez2019": "titulo, ano e preprint (arXiv 1902.06543) conferidos; volume, paginas e lista completa de autores NAO conferidos (entrada com 'and others' e sem volume). Completar na pagina da Medical Image Analysis antes de submeter.",
    "lipton2014f1": "titulo, autores, ECML PKDD 2014, paginas 225--239 e DOI conferidos por busca (arXiv 1402.1892 e registro da Springer). O sobrenome do terceiro autor aparece como 'Narayanaswamy' no arXiv e 'Naryanaswamy' em alguns catalogos: conferir a grafia na pagina da Springer antes de submeter. Trata do limiar que maximiza o F1; texto completo nao lido.",
    "lin1989": "autores, titulo, Biometrics 45(1):255-268 e DOI 10.2307/2532051 conferidos por busca (PubMed e Wikipedia); texto completo nao lido.",
    "bland1986": "autores, titulo, Lancet 1(8476):307-310 conferidos por busca (PubMed); o DOI nao foi conferido e foi omitido; um registro lista o volume como 327 (numeracao atual da Lancet): conferir no PubMed antes de submeter.",
}
assert set(METHODS_NOTES) <= set(meth), set(METHODS_NOTES) - set(meth)
out += [(f"% NOTA ({k}): {METHODS_NOTES[k]}\n" if k in METHODS_NOTES else "") + entries[k] + "\n" for k in meth]
(ART / "referencias.bib").write_text("\n".join(out), encoding="utf-8")

# verificacao do arquivo gerado
chk, _ = parse(ART / "referencias.bib")
rem, _ = parse(ART / "refs_removidas.bib")
txt = (ART / "referencias.bib").read_text(encoding="utf-8")
print("entradas em referencias.bib:", len(chk), "| esperado", len(used), "| iguais:", set(chk) == set(used))
print("chaves duplicadas no arquivo:", len(re.findall(r"@\w+\s*\{", txt)) - len(chk))
print("chaves { } balanceadas:", txt.count("{") == txt.count("}"))
print("removidas:", len(rem), "| sobreposicao com as usadas:", sorted(set(rem) & set(used)) or "nenhuma")
print("nao citadas nem em referencias.bib nem em refs_removidas.bib:", sorted(set(entries) - set(used) - set(rem)) or "nenhuma")
