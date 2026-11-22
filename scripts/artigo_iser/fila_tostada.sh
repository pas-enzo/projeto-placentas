#!/bin/bash
# Fila do teste limpo do Reinhard na TOSTADA (fold test-TOSTADA_val-ROSILHA-M). Uma coisa por vez:
#   1) normaliza os 180 campos do fold com Reinhard (referencia ajustada so nas eguas de treino do fold) e monta as vistas
#      (mantem native_reinhard para voce poder ver as imagens)
#   2) treina micro (YOLO26s) e capilar (YOLO11s) com Reinhard nesse fold e avalia no teste (TOSTADA)
# Parar: criar o arquivo STOP em D:\projeto_placentas_clayton\benchmark_results
PY="D:/miniconda3/envs/PROJETO_PLACENTAS/python.exe"
cd "D:/projeto_placentas_clayton/dev/projeto-placentas" || exit 1
LOG="D:/projeto_placentas_clayton/temp_ml/artigo_iser/fila_tostada_log.txt"
{
echo "=== $(date +%H:%M:%S) 1/2 normalizacao Reinhard no fold da TOSTADA"
"$PY" scripts/data/build_views.py --fold test-TOSTADA_val-ROSILHA-M --norm reinhard --keep-native-norm --resume || { echo "FALHOU na normalizacao"; exit 1; }
echo "=== $(date +%H:%M:%S) 2/2 treino e avaliacao (micro 26s + capilar 11s)"
"$PY" -m benchmark.runner run benchmark/configs/phase6_tostada_reinhard.yaml --results "D:/projeto_placentas_clayton/benchmark_results" || { echo "FALHOU no runner"; exit 1; }
echo "FILA TOSTADA COMPLETA $(date +%H:%M:%S)"
} >> "$LOG" 2>&1
