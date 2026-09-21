$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$uvExecutable = Join-Path $projectRoot '.tools\uv.exe'

if (-not (Test-Path -LiteralPath $uvExecutable)) {
    $toolsDirectory = Join-Path $projectRoot '.tools'
    $uvArchive = Join-Path $toolsDirectory 'uv.zip'
    New-Item -ItemType Directory -Path $toolsDirectory -Force | Out-Null
    & curl.exe --fail --location --retry 2 `
        'https://github.com/astral-sh/uv/releases/download/0.12.10/uv-x86_64-pc-windows-msvc.zip' `
        --output $uvArchive
    if ($LASTEXITCODE -ne 0) { throw 'uv download failed.' }
    Expand-Archive -LiteralPath $uvArchive -DestinationPath $toolsDirectory -Force
}

$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.python'
$env:UV_LINK_MODE = 'copy'
$env:UV_HTTP_TIMEOUT = '180'
$env:UV_NO_PROGRESS = '1'
Push-Location $projectRoot
try {
    $syncArguments = @('sync', '--python', '3.12.14', '--managed-python')
    if (Test-Path -LiteralPath 'uv.lock') { $syncArguments += '--locked' }
    & $uvExecutable @syncArguments
    if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed.' }

    $assetLock = Get-Content -LiteralPath 'assets.lock.json' -Raw -Encoding UTF8 | ConvertFrom-Json
    $modelPath = Join-Path $projectRoot $assetLock.model.path
    if (-not (Test-Path -LiteralPath $modelPath)) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $modelPath) -Force | Out-Null
        $downloadPath = $modelPath + '.download'
        & curl.exe --fail --location --retry 2 $assetLock.model.url --output $downloadPath
        if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
        if ((Get-FileHash -LiteralPath $downloadPath -Algorithm SHA256).Hash -ne $assetLock.model.sha256) {
            throw 'Downloaded model checksum does not match assets.lock.json.'
        }
        Move-Item -LiteralPath $downloadPath -Destination $modelPath
    }
    if ((Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash -ne $assetLock.model.sha256) {
        throw 'Existing model checksum does not match assets.lock.json; file preserved.'
    }

    foreach ($sample in $assetLock.samples) {
        $samplePath = Join-Path $projectRoot $sample.path
        if (-not (Test-Path -LiteralPath $samplePath)) {
            New-Item -ItemType Directory -Path (Split-Path -Parent $samplePath) -Force | Out-Null
            $bundledPath = Join-Path $projectRoot ('.venv\Lib\site-packages\ultralytics\assets\' + (Split-Path -Leaf $samplePath))
            Copy-Item -LiteralPath $bundledPath -Destination $samplePath
        }
        if ((Get-FileHash -LiteralPath $samplePath -Algorithm SHA256).Hash -ne $sample.sha256) {
            throw ('Sample checksum does not match assets.lock.json: ' + $samplePath)
        }
    }
    Write-Output 'Ready: pinned Python environment, YOLO26n weights, and two sample images.'
} finally {
    Pop-Location
}
