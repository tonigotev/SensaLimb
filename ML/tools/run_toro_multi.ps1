param(
    [int]$Runs = 3,
    [int]$StartSeed = 1,
    [string]$ExtraArgs = "--use-file-split --movement all --emg-source filtered --angle-history-sec 0.1 --tflite-int8",
    [switch]$Parallel
)

$scriptPath = "ML\train\train_conv1d_angle_toro_ossaba.py"

for ($i = 0; $i -lt $Runs; $i++) {
    $seed = $StartSeed + $i
    $ts = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $runName = "seed${seed}_$ts"
    $args = "$scriptPath $ExtraArgs --seed $seed --run-name $runName"
    if ($Parallel) {
        Start-Process powershell "-NoExit -Command python $args"
    } else {
        python $args
    }
}
                                                                                                        