#!/usr/bin/env bash
# Espera o fluxo de validacao terminar ("FLUXO COMPLETO" no log) e so entao regenera os campos nativos do fold principal.
LOG=/d/projeto_placentas_clayton/temp_ml/artigo_iser/run1/log.txt
until grep -q "FLUXO COMPLETO" "$LOG"; do sleep 120; done
cd /d/projeto_placentas_clayton/dev/projeto-placentas || exit 1
export PYTHONIOENCODING=utf-8
/d/miniconda3/envs/PROJETO_PLACENTAS/python.exe scripts/artigo_iser/regenerar_nativos.py --verify --methods vahadane
/d/miniconda3/envs/PROJETO_PLACENTAS/python.exe scripts/artigo_iser/regenerar_nativos.py
echo "NATIVOS COMPLETO"
