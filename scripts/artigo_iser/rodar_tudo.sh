#!/usr/bin/env bash
# Fluxo completo (retomavel: rodar de novo continua de onde parou). Parar: criar o arquivo STOP na pasta de saida.
# Ordem: tudo sem Vahadane primeiro (CPU leve), Vahadane por ultimo (~2,4 h so de normalizacao).
cd /d/projeto_placentas_clayton/dev/projeto-placentas || exit 1
export PYTHONIOENCODING=utf-8
PY=/d/miniconda3/envs/PROJETO_PLACENTAS/python.exe
OUT=D:/projeto_placentas_clayton/temp_ml/artigo_iser/run1
F=scripts/artigo_iser/fluxo.py

$PY $F normalize --out $OUT --norms reinhard macenko whitebalance histmatch grayscale || exit 1
$PY $F infer --out $OUT --variants A_none A_reinhard B_reinhard A_macenko B_macenko A_whitebalance B_whitebalance \
    A_histmatch B_histmatch A_grayscale B_grayscale || exit 1
# parcial (sem Vahadane): ja da para olhar enquanto o resto roda
$PY $F report --out $OUT
$PY $F panels --out $OUT
$PY $F normalize --out $OUT --norms vahadane || exit 1
$PY $F infer --out $OUT --variants A_vahadane B_vahadane || exit 1
$PY $F report --out $OUT
$PY $F panels --out $OUT --force
echo "FLUXO COMPLETO"
