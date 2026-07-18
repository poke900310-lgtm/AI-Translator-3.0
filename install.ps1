# install.ps1
# Sets up a fully local environment for this project:
# - Creates .\.venv and installs Python dependencies (Python 3.12+ required)
# - Creates .\models (you provide your own .gguf model file)
# - Detects GPU vendor and downloads the latest llama.cpp binaries:
#     AMD    -> latest ROCm build from repo.radeon.com
#     NVIDIA -> latest CUDA build from github.com/ggerganov/llama.cpp
#     Other  -> latest Vulkan build from github.com/ggerganov/llama.cpp (fallback)
# - Validates that llama-server.exe exists
# - Uses the bundled One OCR runtime already included in this package
#
# Usage (PowerShell):
#   cd <project>
#   .\install.ps1              <- interactive menu to pick backend
#   .\install.ps1 -Backend 0   <- auto-detect GPU (non-interactive)
#   .\install.ps1 -Backend 1   <- force NVIDIA (CUDA)
#   .\install.ps1 -Backend 2   <- force AMD (ROCm)
#   .\install.ps1 -Backend 3   <- force Vulkan / universal
#   .\install.ps1 -Dev         <- additionally install dev tooling (pytest,
#                                 mypy, ruff). Combine with -Backend.
#
# Then run:
#   .\run.bat          <- start the translator
#   .\reset_state.bat  <- wipe debug output and runtime state

param(
  [int]$Backend = -1,  # -1=show menu  0=auto  1=NVIDIA  2=AMD  3=Vulkan
  [switch]$Dev         # also install requirements-dev.txt
)

$ErrorActionPreference = "Stop"

function Write-Section([string]$Title) {
  Write-Host ""
  Write-Host "=== $Title ===" -ForegroundColor Cyan
}

function Require-Command([string]$Cmd, [string]$Hint) {
  if (-not (Get-Command $Cmd -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $Cmd. $Hint"
  }
}

# WebClient is faster than Invoke-WebRequest on Windows PowerShell 5.x
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Download-File([string]$Url, [string]$OutFile) {
  $wc = New-Object System.Net.WebClient
  try {
    $wc.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT; Win64; x64)")
    $wc.DownloadFile($Url, $OutFile)
  } finally {
    $wc.Dispose()
  }
}

function Download-String([string]$Url) {
  $wc = New-Object System.Net.WebClient
  try {
    $wc.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT; Win64; x64)")
    return $wc.DownloadString($Url)
  } finally {
    $wc.Dispose()
  }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$LlamaRootDir = Join-Path $ProjectRoot "llama_cpp\binaries"

# ---- GPU selection ----
Write-Section "GPU backend"

if ($Backend -lt 0) {
  Write-Host ""
  Write-Host "  [0] Auto-detect"
  Write-Host "  [1] NVIDIA  (CUDA)"
  Write-Host "  [2] AMD     (ROCm)"
  Write-Host "  [3] Vulkan  (universal)"
  Write-Host ""
  do {
    $raw = Read-Host "  Select backend (0-3)"
    $parsed = 0
    $ok = [int]::TryParse($raw.Trim(), [ref]$parsed) -and $parsed -ge 0 -and $parsed -le 3
    if (-not $ok) { Write-Warning "Please enter 0, 1, 2, or 3." }
  } while (-not $ok)
  $Backend = $parsed
}

Write-Host "  Backend: $Backend  (0=auto  1=NVIDIA  2=AMD  3=Vulkan)"

if ($Backend -eq 1) {
  $vendor = "nvidia"
  Write-Host "Forced: NVIDIA (CUDA)" -ForegroundColor Cyan
} elseif ($Backend -eq 2) {
  $vendor = "amd"
  Write-Host "Forced: AMD (ROCm)" -ForegroundColor Cyan
} elseif ($Backend -eq 3) {
  $vendor = "vulkan"
  Write-Host "Forced: Vulkan / universal" -ForegroundColor Cyan
} else {
  $gpuObjects = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue
  foreach ($g in $gpuObjects) { Write-Host "  Found: $($g.Name)" }
  $vendor = "unknown"
  foreach ($g in $gpuObjects) {
    if ($g.Name -match "NVIDIA" -or $g.AdapterCompatibility -match "NVIDIA") { $vendor = "nvidia"; break }
  }
  if ($vendor -eq "unknown") {
    foreach ($g in $gpuObjects) {
      if ($g.Name -match "AMD|Radeon" -or $g.AdapterCompatibility -match "AMD|Advanced Micro") { $vendor = "amd"; break }
    }
  }
  Write-Host "Auto-detected: $vendor" -ForegroundColor $(if ($vendor -ne "unknown") { "Green" } else { "Yellow" })
}

# ---- URL resolvers ----
function Get-AmdLlamaUrl {
  # Scrapes repo.radeon.com for the latest ROCm Windows build
  $base = "https://repo.radeon.com/rocm/llama.cpp/windows/"
  try {
    $html = Download-String $base
    $dirs = [regex]::Matches($html, 'href="(rocm-rel-[\d.]+)/"') | ForEach-Object { $_.Groups[1].Value }
    if (-not $dirs) { return $null }
    $latestDir = $dirs | Sort-Object {
      $v = $_ -replace "rocm-rel-", ""
      try { [version]"$v.0" } catch { [version]"0.0.0" }
    } | Select-Object -Last 1
    $subHtml = Download-String "$base$latestDir/"
    $zips = [regex]::Matches($subHtml, 'href="(llama-b\d+-windows-rocm[^"]+x64\.zip)"') |
            ForEach-Object { $_.Groups[1].Value }
    if (-not $zips) { return $null }
    $latestZip = $zips | Sort-Object {
      try { [int]([regex]::Match($_, 'llama-b(\d+)').Groups[1].Value) } catch { 0 }
    } | Select-Object -Last 1
    return "$base$latestDir/$latestZip"
  } catch {
    Write-Warning "AMD URL lookup failed: $_"
    return $null
  }
}

function Get-NvidiaLlamaUrl {
  # Uses GitHub releases API to find the latest CUDA Windows build
  try {
    $json = Download-String "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    $release = $json | ConvertFrom-Json
    $cudaAssets = $release.assets | Where-Object { $_.name -match "bin-win-cuda-cu[\d.]+-x64\.zip" }
    if (-not $cudaAssets) { return $null }
    $best = $cudaAssets | Sort-Object {
      $v = [regex]::Match($_.name, "cu([\d.]+)").Groups[1].Value
      try { [version]"$v.0" } catch { [version]"0.0.0" }
    } | Select-Object -Last 1
    return $best.browser_download_url
  } catch {
    Write-Warning "NVIDIA URL lookup failed: $_"
    return $null
  }
}

function Get-VulkanLlamaUrl {
  # Vulkan build from GitHub — works on AMD, NVIDIA, and Intel
  try {
    $json = Download-String "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    $release = $json | ConvertFrom-Json
    $asset = $release.assets | Where-Object { $_.name -match "bin-win-vulkan-x64\.zip" } | Select-Object -First 1
    if (-not $asset) { return $null }
    return $asset.browser_download_url
  } catch {
    Write-Warning "Vulkan URL lookup failed: $_"
    return $null
  }
}

# ---- Prerequisites ----
Write-Section "Checking prerequisites"
Require-Command "python" "Install Python 3.12+ and ensure 'python' is on PATH."
$pyVersion = & python --version 2>&1
Write-Host "  Detected: $pyVersion"
if ($pyVersion -notmatch "Python 3\.(1[2-9]|[2-9][0-9])") {
  Write-Warning "This project targets Python 3.12 (see pyproject.toml). Older versions may work but are unsupported."
}

function Require-Pip() {
  try {
    $null = python -m pip --version
  } catch {
    throw "Python pip is not available. Reinstall Python with pip enabled, or run: python -m ensurepip --upgrade"
  }
}
Require-Pip

Write-Section "Creating local virtual environment (.venv)"
if (-not (Test-Path ".\.venv")) {
  python -m venv .venv
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  throw "Virtual environment python not found at: $VenvPython"
}

Write-Section "Installing Python dependencies"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($Dev) {
  Write-Section "Installing dev tooling (pytest, mypy, ruff)"
  & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt")
}

Write-Section "Ensuring models folder exists"
$ModelsDir = Join-Path $ProjectRoot "models"
if (-not (Test-Path $ModelsDir)) { New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null }

# ---- llama.cpp binaries ----
if (-not (Test-Path $LlamaRootDir)) { New-Item -ItemType Directory -Force -Path $LlamaRootDir | Out-Null }

$ServerExe = Get-ChildItem -Path $LlamaRootDir -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue | Select-Object -First 1

# Inspect any DLL siblings of llama-server.exe to guess what backend the
# existing binaries are (NVIDIA / AMD / Vulkan). Used purely for the
# mismatch warning below — empty string = unknown.
function Detect-InstalledBackend([string]$ServerDir) {
  if (-not (Test-Path $ServerDir)) { return "" }
  $dllNames = (Get-ChildItem -Path $ServerDir -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name.ToLower() })
  foreach ($n in $dllNames) {
    if ($n -like "cudart*" -or $n -like "cublas*" -or $n -like "ggml-cuda*") { return "nvidia" }
    if ($n -like "amdhip*"  -or $n -like "rocblas*" -or $n -like "ggml-hip*")  { return "amd" }
    if ($n -like "vulkan*"  -or $n -like "ggml-vulkan*")                       { return "vulkan" }
  }
  return ""
}

if ($ServerExe) {
  Write-Section "llama.cpp already installed"
  Write-Host "Found: $($ServerExe.FullName)" -ForegroundColor Green
  $installedBackend = Detect-InstalledBackend $ServerExe.Directory.FullName
  if ($installedBackend -and $vendor -and ($installedBackend -ne $vendor)) {
    Write-Warning ("Existing binaries look like '$installedBackend' but you asked for '$vendor'. " +
                   "To switch backends, delete '$LlamaRootDir' and re-run install.ps1 -Backend N.")
  } elseif ($installedBackend) {
    Write-Host "Detected backend: $installedBackend" -ForegroundColor Gray
  }
  Write-Host "Delete '$LlamaRootDir' and re-run to upgrade or switch backends." -ForegroundColor Gray
} else {
  Write-Section "Resolving llama.cpp download URL"
  $LlamaZipUrl = $null

  if ($vendor -eq "amd") {
    Write-Host "Looking up latest AMD ROCm build..."
    $LlamaZipUrl = Get-AmdLlamaUrl
    if ($LlamaZipUrl) {
      Write-Host "AMD URL: $LlamaZipUrl" -ForegroundColor Green
    } else {
      Write-Warning "Could not resolve AMD URL; will try Vulkan fallback."
    }
  } elseif ($vendor -eq "nvidia") {
    Write-Host "Looking up latest NVIDIA CUDA build..."
    $LlamaZipUrl = Get-NvidiaLlamaUrl
    if ($LlamaZipUrl) {
      Write-Host "NVIDIA URL: $LlamaZipUrl" -ForegroundColor Green
    } else {
      Write-Warning "Could not resolve NVIDIA CUDA URL; will try Vulkan fallback."
    }
  } else {
    # vulkan or unknown — go straight to Vulkan
    Write-Host "Using Vulkan / universal build."
  }

  if (-not $LlamaZipUrl) {
    Write-Host "Looking up Vulkan build..."
    $LlamaZipUrl = Get-VulkanLlamaUrl
    if (-not $LlamaZipUrl) {
      throw "Could not resolve any llama.cpp download URL. Check your internet connection and try again."
    }
    Write-Host "Vulkan URL: $LlamaZipUrl" -ForegroundColor Yellow
  }

  $LlamaZipName = [System.IO.Path]::GetFileName($LlamaZipUrl)
  $LlamaZipPath = Join-Path $ProjectRoot $LlamaZipName

  Write-Section "Downloading llama.cpp binaries"
  if (-not (Test-Path $LlamaZipPath)) {
    Write-Host "Downloading: $LlamaZipUrl"
    Download-File -Url $LlamaZipUrl -OutFile $LlamaZipPath
  } else {
    Write-Host "Zip already present: $LlamaZipPath"
  }

  Write-Host "Extracting to: $LlamaRootDir"
  Expand-Archive -Path $LlamaZipPath -DestinationPath $LlamaRootDir -Force

  $ServerExe = Get-ChildItem -Path $LlamaRootDir -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue | Select-Object -First 1

  if ($ServerExe) {
    Write-Host "Removing extracted zip to save disk space: $LlamaZipPath"
    Remove-Item -Path $LlamaZipPath -Force -ErrorAction SilentlyContinue
  }
}

Write-Section "Locating llama-server.exe"
if (-not $ServerExe) {
  throw "Could not find llama-server.exe under: $LlamaRootDir"
}
Write-Host "Found llama-server.exe: $($ServerExe.FullName)" -ForegroundColor Green

Write-Section "Checking for a GGUF model"
$existingModels = @(Get-ChildItem $ModelsDir -Filter "*.gguf" -ErrorAction SilentlyContinue)
if ($existingModels.Count -gt 0) {
  Write-Host "Found $($existingModels.Count) model file(s) in $ModelsDir :" -ForegroundColor Green
  foreach ($m in $existingModels) {
    Write-Host ("  {0}  ({1:N0} bytes)" -f $m.Name, $m.Length)
  }
  Write-Host "Set LLAMA_SERVER_MODEL_PATH in config.py (or as an env var) to the file you want to use." -ForegroundColor Gray
} else {
  Write-Warning "No .gguf files in $ModelsDir."
  Write-Host "Drop a GGUF model into .\models and set LLAMA_SERVER_MODEL_PATH in app/config.py" -ForegroundColor Yellow
  Write-Host "(default expects: .\models\Sugoi-14B-Ultra-Q4_K_M.gguf)" -ForegroundColor Yellow
}

Write-Section "Verifying app modules import cleanly"
# Catches broken venv / missing native DLL / PyQt6 wheel mismatch BEFORE
# the user clicks run.bat and gets a Qt error dialog blaming the wrong
# layer. Throws if either import fails.
$smokeOutput = & $VenvPython -c "from app import config; from app.controller import Controller; print('Import smoke OK')" 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host $smokeOutput -ForegroundColor Red
  throw "Post-install import smoke failed. The venv is set up but the app modules don't import cleanly - fix the error above before running."
}
Write-Host $smokeOutput -ForegroundColor Green

Write-Section "Done"
Write-Host "Start:        .\run.bat" -ForegroundColor Green
Write-Host "Reset state:  .\reset_state.bat  (clears debug, runtime config, caches)" -ForegroundColor Green
if ($Dev) {
  Write-Host "Run tests:    .\.venv\Scripts\python.exe -m pytest -q" -ForegroundColor Green
  Write-Host "Type check:   .\.venv\Scripts\python.exe -m mypy app/" -ForegroundColor Green
  Write-Host "Lint:         .\.venv\Scripts\python.exe -m ruff check app tests main.py scripts" -ForegroundColor Green
}
