#!/usr/bin/env bash
# Fase 9 (2026-11-24): treina os modelos adotados COM Reinhard nos 5 folds que faltam.
#   1) gera as vistas normalizadas de cada fold (referencia de cor ajustada so com as 4 eguas de treino do fold);
#   2) roda benchmark/configs/phase9_reinhard_seisfolds.yaml (10 treinos: micro YOLO26s e capilar YOLO11s em cada fold).
# Retomavel: rodar de novo continua de onde parou (vistas prontas sao puladas; o runner pula treinos concluidos).
# Parar: criar o arquivo STOP em D:\projeto_placentas_clayton\benchmark_results (termina a execucao atual e sai).
# Log: temp_ml/artigo_iser/fila_reinhard_folds_log.txt
cd /d/projeto_placentas_clayton/dev/projeto-placentas || exit 1
export PYTHONIOENCODING=utf-8
PY=/d/miniconda3/envs/PROJETO_PLACENTAS/python.exe
RES=D:/projeto_placentas_clayton/benchmark_results
STOP=/d/projeto_placentas_clayton/benchmark_results/STOP
FOLDS="test-TOSTADA_val-ROSILHA-M test-ROSILHA-M_val-TORDILHA test-TORDILHA_val-TP test-TP_val-ZAINA-679 test-ZAINA-679_val-ZAZA"

echo "=== $(date '+%d/%m %H:%M:%S') INICIO fila Reinhard nos folds restantes"
for F in $FOLDS; do
  [ -e "$STOP" ] && { echo "STOP encontrado: saindo antes de $F"; exit 0; }
  echo "=== $(date '+%d/%m %H:%M:%S') vistas Reinhard do fold $F"
  $PY -X utf8 scripts/data/build_views.py --fold "$F" --norm reinhard --views micro,capilar --resume || { echo "ERRO nas vistas de $F"; exit 1; }
done

echo "=== $(date '+%d/%m %H:%M:%S') treinos (phase9)"
$PY -X utf8 -m benchmark.runner run benchmark/configs/phase9_reinhard_seisfolds.yaml --results "$RES"
echo "=== $(date '+%d/%m %H:%M:%S') FILA REINHARD COMPLETA"
