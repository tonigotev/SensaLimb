param(
    [int]$Runs = 5,
    [string]$Root = "ML\models\conv1d_angle_toro_ossaba"
)

$base = Resolve-Path $Root
$folders = Get-ChildItem -Path $base -Directory |
    Sort-Object LastWriteTime -Descending

$opened = 0
foreach ($dir in $folders) {
    $img = Join-Path $dir.FullName "pred_vs_true.png"
    if (Test-Path $img) {
        Write-Host ("Opening: " + $dir.Name)
        Start-Process $img
        $opened += 1
        if ($opened -ge $Runs) { break }
    }
}

if ($opened -eq 0) {
    Write-Host "No pred_vs_true.png files found under $base"
}
