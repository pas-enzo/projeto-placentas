# Plano do benchmark (artigo)

Comparar **modelos, técnicas e normalizações de coloração** na segmentação de microcotilédones e capilares, sobre a divisão **sem vazamento entre éguas** (`data/splits/split_v1_por_egua.json`). Este documento reúne as decisões tomadas e serve de especificação do executor.

## 1. Escopo e princípios
- **Tarefas:** (a) microcotilédones, (b) capilares. Cada técnica só entra se produzir **contagem e área por estrutura** (direto por instância, ou por pós-processamento de máscara semântica).
- **Dados:** uma fonte única (`datasat_v3.1_yolo11_og_size`, nativa 4140 × 3096) + manifesto da divisão. As vistas (`micro_stretch640`, `capilar_tiles3x3` e as normalizadas) são **derivadas por script**, nunca editadas à mão.
- **Fold:** triagem no fold fixo (`test-ZAZA_val-TOSTADA`); finalistas nos 6 folds (média ± desvio entre éguas).
- **Seleção na validação, relato no teste.** O teste nunca participa da escolha de modelo, limiar ou época.
- **Reprodutível e retomável:** cada execução grava configuração, versões, hash do manifesto e do git; a fila pode ser interrompida e continuada (seção 7); roda igual em máquina remota.

## 2. Catálogo de técnicas (~45 candidatos por tarefa; ~30 devem sobreviver ao teste de 1 época)
| Tipo | Modelos | Nº |
|---|---|---|
| YOLO por instância (CNN) | YOLOv8 n/s/m · YOLO11 n/s/m/l · YOLO26 n/s/m · YOLOv9-seg · YOLOE-seg · YOLOv12 n/s (se a segmentação funcionar) | ~14 |
| Transformers por instância | RF-DETR Seg Nano/Small/Medium · Mask2Former Swin-T e R50 | ~5 |
| Dois estágios | Mask R-CNN R50-FPN (v1/v2) · *backbone* Swin-T / ConvNeXt-T | ~4 |
| MMDetection (ambiente separado) | RTMDet-Ins tiny/s · SOLOv2 · CondInst | ~4 |
| Semânticos + recuperação de instâncias | U-Net (ResNet34, EfficientNet-B3, ConvNeXt-T) · U-Net++ · FPN · DeepLabV3+ · SegFormer b0/b2 | ~8 |
| Histologia | InstanSeg · Cellpose 3 · Cellpose-SAM (se couber) | ~3 |
| Detector + SAM 2 | YOLO11s e RF-DETR-det com SAM 2 Hiera-Tiny | ~2 |
| Só detecção (contagem; sem área) | RT-DETR · D-FINE · DEIM | ~5 |

Restrição de hardware: GTX 1650 SUPER, 4 GB (~2,7 GB livres). Modelos que não cabem ou não instalam são registrados como `skipped` com o motivo, e não removidos em silêncio.

## 3. Variações
- **Normalização de coloração** (modelo de referência: YOLO11s): sem normalização · balanço de branco (gray-world) · Reinhard (com máscara de tecido) · Macenko · histogram matching · escala de cinza · augmentação de cor HED/RandStainNA (sem vista nova). A referência de cor vem **só das éguas de treino**; a mesma transformação vale para validação e teste, e o pipeline precisa aplicá-la na inferência.
- **Configurações** (modelo de referência): tamanho de entrada, força da augmentação (*mosaic*/*copy-paste*), otimizador, congelar *backbone*, parâmetros de máscara.
- ~13 variações por tarefa.

## 4. Fases e tempo estimado (GTX 1650 SUPER)
Medido: capilar YOLO11s ≈ 3,3 h (60 épocas, 1.080 ladrilhos); micro YOLO11s ≈ 0,7 h (120 épocas). O capilar pesa ~5× mais.

| Fase | Execuções | Tempo |
|---|---|---|
| 1. Triagem: ~30 modelos de micro + ~20 de capilar + variações, fold fixo, 1 semente | ~75 | ~160–170 h (~7 dias) |
| 2. Cruzamento: 5 melhores modelos × 3 melhores normalizações | ~30 | ~60 h |
| 3. Finalistas nos 6 folds | ~36 | ~90 h |

As fases 2 e 3 podem ir para máquina remota.

## 5. Métricas
Todas no **teste**, com o limiar calibrado na **validação**.

| Para quê | Métrica |
|---|---|
| **Ranking (decisão)** | **AP de máscara 50–95** (COCO), sem dependência de limiar; AP50 e AP75 também reportados |
| **Calibrar o limiar de confiança** (validação) | **F1** por instância |
| **Comparação comum a todas as técnicas** (inclusive semânticas) | **Dice** e **IoU** de pixel da união das máscaras (reportadas; calibram o limiar dos modelos semânticos) |
| **Desfecho clínico** (independente da escolha) | erro de área e de contagem por campo, concordância e Bland–Altman |

**Instância:** precisão, revocação e F1 (IoU ≥ 0,5; pareamento guloso por IoU), IoU médio dos pares.

**Morfometria, por campo:**
- erro relativo de **área** em módulo (média e mediana) e o viés com sinal;
- erro de **contagem**;
- correlação de Pearson, **CCC de Lin** e Bland–Altman (viés e limites de concordância);
- fração de área microcotiledonária e fração capilar (capilar ÷ micro).
Área reportada como **soma** das instâncias (definição do protocolo manual) e como **união** (desconta sobreposição).

**Eficiência:** parâmetros, tamanho do arquivo, VRAM de pico, tempo de treino, tempo de inferência por campo (**de ponta a ponta**).

**Robustez:** média ± desvio entre éguas (finalistas), pior égua e quebra por região (B/D/G).

**Estatística:** IC de 95% por *bootstrap* sobre os campos de teste; comparações pareadas contra o baseline com correção de Holm.

### 5.1 Calibração do limiar de confiança
- Critério: **maximizar o F1 na validação** (varredura de 0,05 a 0,95). **Não** usar o S(c) do artigo (pesos arbitrários, e o termo de área agregada é suscetível a compensação entre sobreposição e instâncias perdidas).
- **A área é um desfecho independente, não critério de escolha.**
- Modelos semânticos (sem confiança por instância): limiar de probabilidade calibrado pelo Dice de pixel; a métrica por instância depende do pós-processamento (componentes conexos / *watershed*) e é sinalizada como tal. Para eles o ranking usa o Dice (não existe AP).
- O S(c) segue reportado apenas por continuidade com o artigo.

### 5.2 Avaliação do capilar
Em **campo nativo** para todos os modelos, com **o mesmo preditor por fatias** (1380 × 1032, sobreposição 20%, fusão GREEDYNMM/IOS 0,5), e não o SAHI de cada biblioteca, para a comparação ser justa. A métrica por ladrilho fica só como diagnóstico.

## 6. Protocolo de justiça
- Mesma divisão, mesma semente (triagem), mesmo critério de épocas máximas e parada antecipada **por família**, registrado em cada execução.
- Se faltar memória: nova tentativa com *batch* menor, registrada na execução.
- Famílias que exigem outro formato recebem uma **vista própria** (COCO para RF-DETR, Mask R-CNN, Mask2Former, MMDetection; máscaras semânticas para U-Net etc.), geradas a partir do mesmo manifesto.

## 7. Interrupção e retomada (requisito)
Quero poder **interromper a fila e continuar a partir da última execução concluída por completo**, e, nas famílias que permitem, **retomar no meio do treino**.

**Por execução concluída**
- **Identificador fixo** derivado da configuração (modelo, normalização, config, tarefa, fold, semente): a mesma matriz gera sempre os mesmos identificadores.
- Uma execução só conta como concluída com o marcador **`DONE`**, gravado **por último**, depois de pesos, métricas de validação, métricas de teste e `run.json` estarem completos e válidos. Interrupção no meio (queda de energia, suspensão do Windows, falha da GPU, Ctrl+C) não deixa `DONE`.
- Ao reiniciar, o executor ignora as concluídas, isola os restos de uma execução interrompida e segue a mesma ordem da fila.
- Arquivos de estado gravados de forma **atômica** (temporário + renomear).

**Por etapa:** treino → avaliação na validação → avaliação no teste, cada uma com marcador próprio. Interrupção durante a avaliação refaz só a avaliação (importante no capilar: ~17 min de avaliação em campo contra ~3 h de treino).

**Retomada no meio do treino (famílias que permitem)**
| Família | Mecanismo |
|---|---|
| Ultralytics (YOLO, YOLOE, YOLOv12) | `resume` a partir do `last.pt` (salvo a cada época): restaura modelo, otimizador, EMA e época |
| RF-DETR | `resume` a partir do *checkpoint*; ajustar o intervalo de gravação para ser frequente (a cada 1 a 5 épocas) |
| Mask R-CNN, Mask2Former, semânticos, histologia | laço de treino próprio que grava, a cada época, modelo, otimizador, *scheduler*, época e estado dos geradores aleatórios |
| MMDetection | `--resume` nativo |
| Sem suporte | a execução é refeita desde o início (flag `supports_resume: false` no adaptador) |

Cuidados: a retomada só é permitida se o **hash da configuração** coincide com o da execução original; o resultado fica equivalente, mas **não idêntico bit a bit**, e por isso o `run.json` registra `resumed: true` e a época de retomada.

**Interrupção limpa:** Ctrl+C a qualquer momento é seguro; um arquivo `STOP` faz o executor **terminar a execução atual e sair**.

**Comandos de apoio:** `--status` (concluídas / em andamento / pendentes, com estimativa de tempo restante), `--retry-failed`, filtros para rodar subconjuntos.

**Robustez adicional:** a tabela central `results.csv` é **reconstruída** a partir dos `run.json` (fonte de verdade), e não acrescentada linha a linha; os resultados podem ser copiados para outra máquina e continuados lá; proteção contra disco cheio.

## 8. Estrutura de código prevista
```
benchmark/
  runner.py            fila retomável (matriz → execuções), registro por execução
  configs/*.yaml       matriz de experimentos (modelo × normalização × config × tarefa × fold × semente)
  adapters/            um adaptador por família (ultralytics, rfdetr, torchvision, hf_mask2former, smp, mmdet, ...)
  eval/                preditor por fatias comum, métricas (COCO AP, F1, Dice, morfometria, estatística)
  views/               geração das vistas (COCO, semântica, normalizadas) a partir do manifesto
results/               <run_id>/run.json + marcadores + tabela central results.csv
```
Cada `run.json`: configuração, hash do manifesto e do git, versões, hardware, tempos, VRAM, métricas de validação e teste, `resumed`.

## 9. Instalação
Ambiente principal em D: (`D:\miniconda3\envs\PROJETO_PLACENTAS`) para o que for compatível; **ambientes conda separados em D:** (`bench_*`) para famílias com dependências em conflito (MMDetection, por exemplo), sem afetar o app. Cada instalação é anunciada antes de ser feita.

## 10. Ordem de implementação
1. Exportação COCO das vistas e as normalizações (opção `--norm`).
2. Preditor por fatias comum e módulo de métricas, validados contra os números já conhecidos (YOLO micro e capilar reproduzidos).
3. Executor (com a retomada) e adaptador Ultralytics; teste de 1 época por modelo (tempo e memória reais).
4. Demais adaptadores, em ordem de custo: RF-DETR, Mask R-CNN, Mask2Former, semânticos, histologia, MMDetection, SAM 2.
5. Fila da fase 1.
