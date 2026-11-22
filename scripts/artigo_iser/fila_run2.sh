#!/usr/bin/env bash
# run2 (2026-11-20): o mesmo fluxo do run1, sobre as 150 imagens externas com coloracao boa
# (5 eguas x 3 regioes x 10 campos, copiadas e renomeadas para imagens_teste_artigo_iser_v2; barra de 50 um = 466 px).
# Retomavel: rodar de novo continua de onde parou. Parar: criar o arquivo STOP em run2.
# Vahadane (CPU, a etapa mais lenta) roda em paralelo com a inferencia na GPU das outras variantes.
cd /d/projeto_placentas_clayton/dev/projeto-placentas || exit 1
export PYTHONIOENCODING=utf-8
export ISER_IN_DIR='D:\projeto_placentas_clayton\imagens_teste_artigo_iser_v2'
export ISER_OUT='D:\projeto_placentas_clayton\temp_ml\artigo_iser\run2'
OUT=/d/projeto_placentas_clayton/temp_ml/artigo_iser/run2
OUTW=D:/projeto_placentas_clayton/temp_ml/artigo_iser/run2
PY=/d/miniconda3/envs/PROJETO_PLACENTAS/python.exe
F=scripts/artigo_iser/fluxo.py
mkdir -p "$OUT"
step() { [ -e "$OUT/STOP" ] && { echo "STOP: pulando $1"; return 1; }; echo "=== $(date '+%d/%m %H:%M:%S') $1"; return 0; }

step "0 Vahadane em paralelo (CPU), log em run2/vahadane_log.txt" && \
  ( $PY $F normalize --out $OUTW --norms vahadane > "$OUT/vahadane_log.txt" 2>&1; echo "VAHADANE FIM $(date '+%H:%M:%S')" >> "$OUT/vahadane_log.txt" ) &
VPID=$!

step "1/6 normalizacoes rapidas + inferencia sem Vahadane (11 variantes)" && {
  $PY $F normalize --out $OUTW --norms reinhard macenko whitebalance histmatch grayscale || exit 1
  $PY $F infer --out $OUTW --variants A_none A_reinhard B_reinhard A_macenko B_macenko A_whitebalance B_whitebalance \
      A_histmatch B_histmatch A_grayscale B_grayscale || exit 1; }

step "2/6 saidas do pipeline das 11 variantes (CPU) enquanto o Vahadane termina" && \
  $PY scripts/artigo_iser/exportar_pipeline.py --out $OUTW --workers 4

echo "=== $(date '+%d/%m %H:%M:%S') esperando o Vahadane terminar"
wait $VPID
step "3/6 inferencia A/B_vahadane" && {
  $PY $F normalize --out $OUTW --norms vahadane || exit 1     # confere que nada ficou faltando e refaz cor.csv
  $PY $F infer --out $OUTW --variants A_vahadane B_vahadane || exit 1; }

step "4/6 saidas do pipeline das variantes Vahadane" && \
  $PY scripts/artigo_iser/exportar_pipeline.py --out $OUTW --workers 4 --variants A_vahadane B_vahadane

step "5/6 relatorio" && $PY $F report --out $OUTW

step "6/6 folhas comparativas e folhas por classe" && {
  $PY $F panels --out $OUTW --force
  $PY scripts/artigo_iser/folhas_por_classe.py --out $OUTW; }
echo "FILA RUN2 COMPLETA $(date '+%d/%m %H:%M:%S')"
