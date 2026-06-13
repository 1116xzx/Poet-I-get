param(
    [Parameter(Mandatory = $true)]
    [int]$PidToWatch,
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [Parameter(Mandatory = $true)]
    [string]$MetricsPath,
    [int]$IntervalSeconds = 1800
)

$ErrorActionPreference = "SilentlyContinue"
$logDir = Split-Path -Parent $LogPath
if ($logDir) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

while ($true) {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $process = Get-Process -Id $PidToWatch -ErrorAction SilentlyContinue
    if ($process) {
        $latestMetric = ""
        if (Test-Path $MetricsPath) {
            $latestMetric = Get-Content $MetricsPath -Tail 1
        }
        Add-Content -Path $LogPath -Value "[$now] running pid=$PidToWatch latest_metric=$latestMetric"
        Start-Sleep -Seconds $IntervalSeconds
    } else {
        $latestMetric = ""
        if (Test-Path $MetricsPath) {
            $latestMetric = Get-Content $MetricsPath -Tail 1
        }
        Add-Content -Path $LogPath -Value "[$now] stopped pid=$PidToWatch latest_metric=$latestMetric"
        break
    }
}
