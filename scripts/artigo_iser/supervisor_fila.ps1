# Mantem a fila Reinhard (fase 9) rodando ate concluir os 10 treinos; so para se existir o arquivo STOP.
#  - relanca sozinho se a fila cair/sair antes de terminar (o runner e retomavel);
#  - se o Ultralytics reduzir o lote por OOM ("Reducing to batch"), a run fica invalida (lote != 8 dos baselines):
#    mata a fila, manda a run incompleta para benchmark_quarentena e relanca (espera mais apos falhas seguidas).
$res  = 'D:\projeto_placentas_clayton\benchmark_results'
$quar = 'D:\projeto_placentas_clayton\benchmark_quarentena'
$stop = Join-Path $res 'STOP'
$log  = 'D:\projeto_placentas_clayton\temp_ml\artigo_iser\fila_reinhard_folds_log.txt'
$bash = 'D:\GIT\Git\usr\bin\bash.exe'
$cmd  = 'cd /d/projeto_placentas_clayton/dev/projeto-placentas && bash scripts/artigo_iser/fila_reinhard_folds.sh >> /d/projeto_placentas_clayton/temp_ml/artigo_iser/fila_reinhard_folds_log.txt 2>&1'
# expandable_segments nao e suportado no Windows; estas opcoes reduzem a fragmentacao sem alterar o treino
$env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:128,garbage_collection_threshold:0.6'
New-Item -ItemType Directory -Force $quar | Out-Null

$slog = 'D:\projeto_placentas_clayton\temp_ml\artigo_iser\supervisor_log.txt'
function Say($m) {
    $gpu = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null)
    $linha = (Get-Date -Format 'dd/MM HH:mm:ss') + " [GPU em uso: $gpu] " + $m + "`r`n"
    try {
        # abre com FileShare.ReadWrite: monitores (tail) abertos no arquivo nao podem bloquear a escrita
        $fs = New-Object System.IO.FileStream($slog, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete)
        $b = [System.Text.Encoding]::UTF8.GetBytes($linha)
        $fs.Write($b, 0, $b.Length); $fs.Close()
    } catch { }
}
function Get-Concluidos {
    $n = 0
    foreach ($d in Get-ChildItem $res -Directory -Filter '*__reinhard__test-*' -ErrorAction SilentlyContinue) {
        if (Test-Path (Join-Path $d.FullName 'DONE')) { $n++ }
    }
    return $n
}
function Get-ProcFila {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(bash|python)\.exe$' -and $_.CommandLine -match 'fila_reinhard_folds\.sh|benchmark\.runner run' }
}
function Get-NReducoes { (Select-String -Path $log -Pattern 'Reducing to batch' -Encoding utf8 -ErrorAction SilentlyContinue | Measure-Object).Count }

$falhasSeguidas = 0
$baseReducoes = Get-NReducoes
while ($true) {
    if (Test-Path $stop) { Say 'STOP encontrado, encerrando'; break }
    if ((Get-Concluidos) -ge 10) { Say '10 treinos Reinhard concluidos, encerrando'; break }

    $procs = @(Get-ProcFila)
    if ($procs.Count -eq 0) {
        Say ("fila parada, relancando (concluidos: " + (Get-Concluidos) + "/10)")
        $baseReducoes = Get-NReducoes
        Start-Process -FilePath $bash -ArgumentList @('-c', ('"' + $cmd + '"')) -WindowStyle Minimized | Out-Null
        Start-Sleep -Seconds 30
        continue
    }

    if ((Get-NReducoes) -gt $baseReducoes) {
        $falhasSeguidas++
        Say "OOM: lote reduzido, run invalida. Matando a fila (falha $falhasSeguidas)"
        foreach ($p in $procs) { & taskkill /PID $p.ProcessId /T /F | Out-Null }
        Start-Sleep -Seconds 10
        foreach ($d in Get-ChildItem $res -Directory -Filter '*__reinhard__test-*' -ErrorAction SilentlyContinue) {
            if (-not (Test-Path (Join-Path $d.FullName 'DONE'))) {
                $dest = Join-Path $quar ($d.Name + '__' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
                Move-Item $d.FullName $dest
                Say ("run incompleta movida para " + $dest)
            }
        }
        $espera = if ($falhasSeguidas -ge 3) { 600 } else { 60 }
        Say "aguardando $espera s antes de relancar"
        Start-Sleep -Seconds $espera
        continue
    }
    Start-Sleep -Seconds 15
}
