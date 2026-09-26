# Tile YOLO-seg dataset (3x3) — PowerShell runner (no Python required).
# Mirrors aux_scripts/tile_yolo_dataset.py logic.

param(
  [string]$Src = "D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size",
  [string]$Dst = "D:\projeto_placentas_clayton\dataset_v3.1_yolo11_tiled_3x3",
  [int]$GridX = 3,
  [int]$GridY = 3,
  [double]$MinAreaFrac = 0.15,
  [double]$MinAreaPx = 64.0,
  [long]$JpegQuality = 95
)

Add-Type -AssemblyName System.Drawing
$ErrorActionPreference = "Stop"

function Shoelace([System.Collections.Generic.List[double]]$xs, [System.Collections.Generic.List[double]]$ys) {
  $n = $xs.Count
  if ($n -lt 3) { return 0.0 }
  $a = 0.0
  for ($i = 0; $i -lt $n; $i++) {
    $j = ($i + 1) % $n
    $a += $xs[$i] * $ys[$j] - $xs[$j] * $ys[$i]
  }
  return [Math]::Abs($a) / 2.0
}

function ClipEdge(
  [System.Collections.Generic.List[double]]$xs,
  [System.Collections.Generic.List[double]]$ys,
  [double]$xMin, [double]$yMin, [double]$xMax, [double]$yMax,
  [string]$edge
) {
  $ox = New-Object 'System.Collections.Generic.List[double]'
  $oy = New-Object 'System.Collections.Generic.List[double]'
  $n = $xs.Count
  if ($n -eq 0) { return @($ox, $oy) }

  $px = $xs[$n - 1]
  $py = $ys[$n - 1]
  switch ($edge) {
    "left"  { $prevIn = ($px -ge $xMin) }
    "right" { $prevIn = ($px -le $xMax) }
    "top"   { $prevIn = ($py -ge $yMin) }
    default { $prevIn = ($py -le $yMax) }
  }

  for ($i = 0; $i -lt $n; $i++) {
    $cx = $xs[$i]
    $cy = $ys[$i]
    switch ($edge) {
      "left"  { $curIn = ($cx -ge $xMin) }
      "right" { $curIn = ($cx -le $xMax) }
      "top"   { $curIn = ($cy -ge $yMin) }
      default { $curIn = ($cy -le $yMax) }
    }

    if ($curIn) {
      if (-not $prevIn) {
        $dx = $cx - $px
        $dy = $cy - $py
        if ($edge -eq "left") {
          $t = if ([Math]::Abs($dx) -lt 1e-12) { 0.0 } else { ($xMin - $px) / $dx }
          $ox.Add($xMin); $oy.Add($py + $t * $dy)
        } elseif ($edge -eq "right") {
          $t = if ([Math]::Abs($dx) -lt 1e-12) { 0.0 } else { ($xMax - $px) / $dx }
          $ox.Add($xMax); $oy.Add($py + $t * $dy)
        } elseif ($edge -eq "top") {
          $t = if ([Math]::Abs($dy) -lt 1e-12) { 0.0 } else { ($yMin - $py) / $dy }
          $ox.Add($px + $t * $dx); $oy.Add($yMin)
        } else {
          $t = if ([Math]::Abs($dy) -lt 1e-12) { 0.0 } else { ($yMax - $py) / $dy }
          $ox.Add($px + $t * $dx); $oy.Add($yMax)
        }
      }
      $ox.Add($cx); $oy.Add($cy)
    } elseif ($prevIn) {
      $dx = $cx - $px
      $dy = $cy - $py
      if ($edge -eq "left") {
        $t = if ([Math]::Abs($dx) -lt 1e-12) { 0.0 } else { ($xMin - $px) / $dx }
        $ox.Add($xMin); $oy.Add($py + $t * $dy)
      } elseif ($edge -eq "right") {
        $t = if ([Math]::Abs($dx) -lt 1e-12) { 0.0 } else { ($xMax - $px) / $dx }
        $ox.Add($xMax); $oy.Add($py + $t * $dy)
      } elseif ($edge -eq "top") {
        $t = if ([Math]::Abs($dy) -lt 1e-12) { 0.0 } else { ($yMin - $py) / $dy }
        $ox.Add($px + $t * $dx); $oy.Add($yMin)
      } else {
        $t = if ([Math]::Abs($dy) -lt 1e-12) { 0.0 } else { ($yMax - $py) / $dy }
        $ox.Add($px + $t * $dx); $oy.Add($yMax)
      }
    }
    $px = $cx; $py = $cy; $prevIn = $curIn
  }
  return @($ox, $oy)
}

function ClipPolyToRect(
  [System.Collections.Generic.List[double]]$xs,
  [System.Collections.Generic.List[double]]$ys,
  [double]$xMin, [double]$yMin, [double]$xMax, [double]$yMax
) {
  foreach ($edge in @("left", "right", "top", "bottom")) {
    $res = ClipEdge $xs $ys $xMin $yMin $xMax $yMax $edge
    $xs = $res[0]
    $ys = $res[1]
    if ($xs.Count -eq 0) { return @($xs, $ys) }
  }

  $cx = New-Object 'System.Collections.Generic.List[double]'
  $cy = New-Object 'System.Collections.Generic.List[double]'
  for ($i = 0; $i -lt $xs.Count; $i++) {
    if ($cx.Count -eq 0 -or [Math]::Abs($cx[$cx.Count - 1] - $xs[$i]) -gt 1e-6 -or [Math]::Abs($cy[$cy.Count - 1] - $ys[$i]) -gt 1e-6) {
      $cx.Add($xs[$i]); $cy.Add($ys[$i])
    }
  }
  if ($cx.Count -ge 2 -and [Math]::Abs($cx[0] - $cx[$cx.Count - 1]) -lt 1e-6 -and [Math]::Abs($cy[0] - $cy[$cy.Count - 1]) -lt 1e-6) {
    $cx.RemoveAt($cx.Count - 1)
    $cy.RemoveAt($cy.Count - 1)
  }
  return @($cx, $cy)
}

function Get-JpegEncoder {
  foreach ($c in [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders()) {
    if ($c.MimeType -eq "image/jpeg") { return $c }
  }
  throw "JPEG encoder not found"
}

$jpegCodec = Get-JpegEncoder
$encParams = New-Object System.Drawing.Imaging.EncoderParameters(1)
$encParams.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter(
  [System.Drawing.Imaging.Encoder]::Quality, $JpegQuality
)

if (-not (Test-Path $Src)) { throw "Source not found: $Src" }
if (Test-Path $Dst) { Remove-Item $Dst -Recurse -Force }
New-Item -ItemType Directory -Path $Dst | Out-Null

$stats = [ordered]@{
  images_in = 0
  tiles_out = 0
  empty_tiles = 0
  polys_in = 0
  polys_kept = 0
  polys_dropped_small = 0
  polys_dropped_no_intersect = 0
  area_in_0 = 0.0
  area_in_1 = 0.0
  area_out_0 = 0.0
  area_out_1 = 0.0
  tile_w = 0
  tile_h = 0
}

foreach ($split in @("train", "valid", "test")) {
  $imgDir = Join-Path $Src "$split\images"
  $lblDir = Join-Path $Src "$split\labels"
  if (-not (Test-Path $imgDir)) { continue }

  $outImg = Join-Path $Dst "$split\images"
  $outLbl = Join-Path $Dst "$split\labels"
  New-Item -ItemType Directory -Path $outImg -Force | Out-Null
  New-Item -ItemType Directory -Path $outLbl -Force | Out-Null

  $images = Get-ChildItem $imgDir -File | Where-Object { $_.Extension -match '\.(jpg|jpeg|png|bmp)$' }
  foreach ($imgFile in $images) {
    $stats.images_in++
    $bmp = [System.Drawing.Bitmap]::FromFile($imgFile.FullName)
    try {
      $W = $bmp.Width
      $H = $bmp.Height
      if (($W % $GridX) -ne 0 -or ($H % $GridY) -ne 0) {
        throw "Image $($imgFile.Name) size ${W}x${H} not divisible by ${GridX}x${GridY}"
      }
      $tileW = [int]($W / $GridX)
      $tileH = [int]($H / $GridY)
      $stats.tile_w = $tileW
      $stats.tile_h = $tileH

      $lblPath = Join-Path $lblDir ($imgFile.BaseName + ".txt")
      $polys = New-Object System.Collections.ArrayList
      if (Test-Path $lblPath) {
        foreach ($line in [System.IO.File]::ReadAllLines($lblPath)) {
          if ([string]::IsNullOrWhiteSpace($line)) { continue }
          $t = $line.Split(' ', [StringSplitOptions]::RemoveEmptyEntries)
          if ($t.Count -lt 7 -or (($t.Count - 1) % 2) -ne 0) { continue }
          $cls = [int][double]$t[0]
          $xs = New-Object 'System.Collections.Generic.List[double]'
          $ys = New-Object 'System.Collections.Generic.List[double]'
          for ($i = 1; $i -lt $t.Count; $i += 2) {
            $xs.Add([double]$t[$i] * $W)
            $ys.Add([double]$t[$i + 1] * $H)
          }
          $area = Shoelace $xs $ys
          $stats.polys_in++
          if ($cls -eq 0) { $stats.area_in_0 += $area } else { $stats.area_in_1 += $area }
          [void]$polys.Add(@{ cls = $cls; xs = $xs; ys = $ys; area = $area })
        }
      }

      $stem = $imgFile.BaseName
      for ($ty = 0; $ty -lt $GridY; $ty++) {
        for ($tx = 0; $tx -lt $GridX; $tx++) {
          $x0 = $tx * $tileW
          $y0 = $ty * $tileH
          $rect = New-Object System.Drawing.Rectangle($x0, $y0, $tileW, $tileH)
          $tileBmp = $bmp.Clone($rect, $bmp.PixelFormat)
          $tileName = "{0}_r{1}c{2}" -f $stem, $ty, $tx
          $tileImgPath = Join-Path $outImg ($tileName + ".jpg")
          $tileLblPath = Join-Path $outLbl ($tileName + ".txt")

          $kept = New-Object System.Collections.Generic.List[string]
          foreach ($poly in $polys) {
            $clip = ClipPolyToRect $poly.xs $poly.ys $x0 $y0 ($x0 + $tileW) ($y0 + $tileH)
            $cxs = $clip[0]
            $cys = $clip[1]
            if ($cxs.Count -lt 3) {
              $stats.polys_dropped_no_intersect++
              continue
            }
            $lxs = New-Object 'System.Collections.Generic.List[double]'
            $lys = New-Object 'System.Collections.Generic.List[double]'
            for ($i = 0; $i -lt $cxs.Count; $i++) {
              $lxs.Add($cxs[$i] - $x0)
              $lys.Add($cys[$i] - $y0)
            }
            $areaClip = Shoelace $lxs $lys
            if ($areaClip -lt $MinAreaPx -or $areaClip -lt ($MinAreaFrac * $poly.area)) {
              $stats.polys_dropped_small++
              continue
            }
            $parts = New-Object System.Collections.Generic.List[string]
            $parts.Add([string]$poly.cls)
            $inv = [System.Globalization.CultureInfo]::InvariantCulture
            for ($i = 0; $i -lt $lxs.Count; $i++) {
              $xn = [Math]::Min(1.0, [Math]::Max(0.0, $lxs[$i] / $tileW))
              $yn = [Math]::Min(1.0, [Math]::Max(0.0, $lys[$i] / $tileH))
              # Always use '.' decimal separator (pt-BR locale would emit commas and break YOLO)
              $parts.Add($xn.ToString("F10", $inv))
              $parts.Add($yn.ToString("F10", $inv))
            }
            $kept.Add(($parts -join " "))
            $stats.polys_kept++
            if ($poly.cls -eq 0) { $stats.area_out_0 += $areaClip } else { $stats.area_out_1 += $areaClip }
          }

          $tileBmp.Save($tileImgPath, $jpegCodec, $encParams)
          $tileBmp.Dispose()
          if ($kept.Count -gt 0) {
            [System.IO.File]::WriteAllLines($tileLblPath, $kept)
          } else {
            [System.IO.File]::WriteAllText($tileLblPath, "")
            $stats.empty_tiles++
          }
          $stats.tiles_out++
        }
      }
    }
    finally {
      $bmp.Dispose()
    }

    if (($stats.images_in % 10) -eq 0) {
      Write-Host ("processed {0} source images..." -f $stats.images_in)
    }
  }
}

$yaml = @"
path: $($Dst -replace '\\','/')
train: train/images
val: valid/images
nc: 2
names:
  0: Capilar
  1: microcotiledone
"@
Set-Content -Path (Join-Path $Dst "data.yaml") -Value $yaml -Encoding UTF8

$rel0 = if ($stats.area_in_0 -gt 0) { [Math]::Abs($stats.area_out_0 - $stats.area_in_0) / $stats.area_in_0 } else { $null }
$rel1 = if ($stats.area_in_1 -gt 0) { [Math]::Abs($stats.area_out_1 - $stats.area_in_1) / $stats.area_in_1 } else { $null }

$report = [ordered]@{
  src = $Src
  dst = $Dst
  grid = "$GridX x $GridY"
  tile_size = "$($stats.tile_w)x$($stats.tile_h)"
  min_area_frac = $MinAreaFrac
  min_area_px = $MinAreaPx
  images_in = $stats.images_in
  tiles_out = $stats.tiles_out
  empty_tiles = $stats.empty_tiles
  polys_in = $stats.polys_in
  polys_kept = $stats.polys_kept
  polys_dropped_small = $stats.polys_dropped_small
  polys_dropped_no_intersect = $stats.polys_dropped_no_intersect
  area_capilar_in = [Math]::Round($stats.area_in_0, 2)
  area_capilar_out = [Math]::Round($stats.area_out_0, 2)
  area_capilar_rel_loss = if ($null -ne $rel0) { [Math]::Round($rel0, 6) } else { $null }
  area_micro_in = [Math]::Round($stats.area_in_1, 2)
  area_micro_out = [Math]::Round($stats.area_out_1, 2)
  area_micro_rel_loss = if ($null -ne $rel1) { [Math]::Round($rel1, 6) } else { $null }
}
$report | ConvertTo-Json | Set-Content (Join-Path $Dst "tiling_report.json") -Encoding UTF8
$report.GetEnumerator() | ForEach-Object { "{0} = {1}" -f $_.Key, $_.Value }
