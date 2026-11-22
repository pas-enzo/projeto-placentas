#!/usr/bin/env bash
# Fila unica, UM trabalho pesado por vez. Retomavel: rodar de novo continua de onde parou (cada etapa pula o que ja existe).
# Parar: criar o arquivo STOP em run1 (cada etapa confere antes de comecar e as longas conferem a cada imagem).
OUT=/d/projeto_placentas_clayton/temp_ml/artigo_iser/run1
OUTW=D:/projeto_placentas_clayton/temp_ml/artigo_iser/run1
cd /d/projeto_placentas_clayton/dev/projeto-placentas || exit 1
export PYTHONIOENCODING=utf-8
PY=/d/miniconda3/envs/PROJETO_PLACENTAS/python.exe
F=scripts/artigo_iser/fluxo.py
step() { [ -e "$OUT/STOP" ] && { echo "STOP: pulando $1"; return 1; }; echo "=== $(date '+%H:%M:%S') $1"; return 0; }

# 0) espera o exportador atual (9 variantes) terminar
until [ "$(grep -c '^PRONTO' $OUT/exportar_log.txt)" -ge 9 ]; do sleep 60; done
rm -f "$OUT/STOP"

step "1/7 inferencia que falta, sem Vahadane (B_histmatch, A/B_grayscale)" && {
  $PY $F normalize --out $OUTW --norms reinhard macenko whitebalance histmatch grayscale
  $PY $F infer --out $OUTW --variants B_histmatch A_grayscale B_grayscale; }
step "2/7 imagens do pipeline das variantes novas + relatorio e folhas parciais" && {
  $PY scripts/artigo_iser/exportar_pipeline.py --out $OUTW --workers 3
  $PY $F report --out $OUTW; $PY $F panels --out $OUTW; }
step "3/7 campos NATIVOS normalizados do fold principal (metodos rapidos)" && \
  $PY scripts/artigo_iser/regenerar_nativos.py --methods whitebalance histmatch grayscale reinhard macenko
step "4/7 Vahadane nas 90 imagens novas + inferencia A/B_vahadane" && {
  $PY $F normalize --out $OUTW --norms vahadane
  $PY $F infer --out $OUTW --variants A_vahadane B_vahadane; }
step "5/7 campos NATIVOS normalizados com Vahadane (180 campos, ~4,4 h)" && {
  $PY scripts/artigo_iser/regenerar_nativos.py --verify --methods vahadane
  $PY scripts/artigo_iser/regenerar_nativos.py --methods vahadane; }
step "6/7 imagens do pipeline das variantes Vahadane" && \
  $PY scripts/artigo_iser/exportar_pipeline.py --out $OUTW --workers 3
step "7/7 relatorio final e folhas completas" && {
  $PY $F report --out $OUTW; $PY $F panels --out $OUTW --force; }
echo "FILA FINAL COMPLETA $(date '+%H:%M:%S')"
