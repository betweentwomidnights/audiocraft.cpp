$AcRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PATH = "$AcRoot\build\bin\Release;$env:PATH"
$env:AC_MODELS_DIR = "$AcRoot\models"
Write-Host "[ac] environment ready (AC_MODELS_DIR=$env:AC_MODELS_DIR)"
