#!/usr/bin/env bash
# Lamina anotada x lamina inferida (180 laminas, todas as variantes). Retomavel: rodar de novo continua de onde parou.
# Parar: criar o arquivo STOP em D:/projeto_placentas_clayton/temp_ml/artigo_iser/dataset_vs_modelo
cd /d/projeto_placentas_clayton/dev/projeto-placentas || exit 1
export PYTHONIOENCODING=utf-8
PY=/d/miniconda3/envs/PROJETO_PLACENTAS/python.exe
S=scripts/artigo_iser/dataset_vs_modelo.py
OUT=/d/projeto_placentas_clayton/temp_ml/artigo_iser/dataset_vs_modelo
stop() { [ -e "$OUT/STOP" ] && { echo "STOP: encerrando"; exit 0; }; }

$PY $S gt
echo "=== fase 1: A_none, A_reinhard, B_reinhard e OOF_none"; stop
$PY $S infer --variants A_none A_reinhard B_reinhard OOF_none
$PY $S report; $PY $S imagens --variants A_none A_reinhard
echo "=== fase 2: demais variantes sem Vahadane"; stop
$PY $S infer --variants A_macenko B_macenko A_whitebalance B_whitebalance A_histmatch B_histmatch A_grayscale B_grayscale
$PY $S report
echo "=== fase 3: variantes Vahadane (so rodam se native_vahadane estiver completo)"; stop
$PY $S infer --variants A_vahadane B_vahadane
$PY $S report
echo "DATASET_VS_MODELO COMPLETO $(date '+%H:%M:%S')"
