param(
    [Parameter(Mandatory = $true)][string[]]$Tickers,
    [Parameter(Mandatory = $true)][string]$Start,
    [Parameter(Mandatory = $true)][string]$End,
    [switch]$SkipDashboard
)

$ErrorActionPreference = "Stop"

& python run_pipeline.py --tickers @Tickers --start $Start --end $End --force-collect
if ($LASTEXITCODE -ne 0) {
    throw "Pipeline failed with exit code $LASTEXITCODE"
}

if (-not $SkipDashboard) {
    Write-Host "`nLaunching Streamlit dashboard. Press Ctrl+C to stop it." -ForegroundColor Green
    & python -m streamlit run app.py
}
