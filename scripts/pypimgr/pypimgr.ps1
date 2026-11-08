# ======================================================================
#  pypimgr.ps1  -  ModifiedNEAT PyPI Build & Upload Manager  (Windows)
#  Requirements: PowerShell 5.1+, Python 3.11+; node+npm or bun (dashboard build)
#  Usage: powershell -ExecutionPolicy Bypass -File pypimgr.ps1
#
#  Safety model: builds from a CLEAN STAGING COPY (no src/node_modules/.env/keys/
#  tests/scripts), ships only dashboard/dist, inspects every artifact with
#  verify_dist.py before upload, and keeps the token in memory only.
# ======================================================================

#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$SCRIPT_DIR  = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = if ($env:PROJECT_DIR) { $env:PROJECT_DIR } else { (Resolve-Path (Join-Path $SCRIPT_DIR '..\..')).Path }
$DASH_DIR    = Join-Path $PROJECT_DIR 'ModifiedNEAT\dashboard'
$CACHE_DIR   = Join-Path $SCRIPT_DIR 'pypimgr_cache'
$VENV_DIR    = Join-Path $CACHE_DIR 'venv'
$VENV_PY     = Join-Path $VENV_DIR 'Scripts\python.exe'
$ART_DIR     = Join-Path $CACHE_DIR 'artifacts'
$VERIFY_PY   = Join-Path $SCRIPT_DIR 'verify_dist.py'
$TESTPYPI_URL = 'https://test.pypi.org/legacy/'

# ------ UI helpers ---------------------------------------------------------------
function clr($t, $c) { Write-Host $t -ForegroundColor $c -NoNewline }
function Write-Hr    { Write-Host ('-' * 72) -ForegroundColor DarkGray }
function Write-Ok($m)   { Write-Host '  ' -NoNewline; clr '[OK]' 'Green';  Write-Host "  $m" }
function Write-Warn($m) { Write-Host '  ' -NoNewline; clr '[!]'  'Yellow'; Write-Host "  $m" }
function Write-Err($m)  { Write-Host '  ' -NoNewline; clr '[X]'  'Red';    Write-Host "  $m" }
function Write-Info($m) { Write-Host '  ' -NoNewline; clr '[*]'  'Blue';   Write-Host "  $m" }
function Wait-AnyKey { Write-Host ''; Write-Host '  Press any key to continue...' -ForegroundColor DarkGray -NoNewline; [void][Console]::ReadKey($true); Write-Host '' }
function Confirm-Yes($q) { $r = Read-Host "  ? $q [y/N]"; return ($r -match '^(y|yes)$') }
function Write-Banner($t) { Write-Host ''; Write-Hr; Write-Host "  $t" -ForegroundColor Cyan; Write-Hr }

# ------ Helpers ------------------------------------------------------------------
function Test-Native { param([string]$Name) return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# Runs a native command; returns $true on exit code 0 (never throws on non-zero).
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments = @(), [string]$WorkDir = $null)
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    if ($WorkDir) { Push-Location $WorkDir }
    try { & $Exe @Arguments | Out-Host; return ($LASTEXITCODE -eq 0) }
    finally { if ($WorkDir) { Pop-Location }; $ErrorActionPreference = $prev }
}

function Find-Python {
    $cands = @()
    if ($env:PYTHON) { $cands += ,@($env:PYTHON) }
    $cands += ,@('py', '-3.13'); $cands += ,@('py', '-3.12'); $cands += ,@('py', '-3.11')
    $cands += ,@('python3'); $cands += ,@('python')
    foreach ($c in $cands) {
        if (-not (Test-Native $c[0])) { continue }
        $extra = @(); if ($c.Count -gt 1) { $extra = $c[1..($c.Count - 1)] }
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & $c[0] @extra -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>$null
        $good = ($LASTEXITCODE -eq 0); $ErrorActionPreference = $prev
        if ($good) { return ,$c }
    }
    return $null
}

function Get-Version {
    $m = Select-String -Path (Join-Path $PROJECT_DIR 'setup.py') -Pattern '^\s*version="([^"]+)"' | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value } else { return '?' }
}

function Ensure-Venv {
    $found = Find-Python
    if (-not $found) { Write-Err 'Python 3.11+ not found (set $env:PYTHON)'; return $false }
    $c = $found; $extra = @(); if ($c.Count -gt 1) { $extra = $c[1..($c.Count - 1)] }
    New-Item -ItemType Directory -Force -Path $CACHE_DIR, $ART_DIR | Out-Null
    if (-not (Test-Path $VENV_PY)) {
        Write-Info 'Creating isolated build environment...'
        if (-not (Invoke-Native $c[0] ($extra + @('-m', 'venv', $VENV_DIR)))) { Write-Err 'venv creation failed'; return $false }
    }
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & $VENV_PY -c 'import build, twine' 2>$null; $have = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prev
    if (-not $have) {
        Write-Info 'Installing build + twine...'
        if (-not (Invoke-Native $VENV_PY @('-m', 'pip', 'install', '--quiet', '--upgrade', 'pip', 'build', 'twine', 'setuptools', 'wheel'))) {
            Write-Err 'pip install failed'; return $false
        }
    }
    return $true
}

function Test-DistReady { return (Test-Path (Join-Path $DASH_DIR 'dist\index.html')) }
function Get-Artifacts { if (Test-Path $ART_DIR) { return @(Get-ChildItem $ART_DIR -File | Where-Object { $_.Name -match '\.(whl|tar\.gz)$' } | ForEach-Object { $_.FullName }) } else { return @() } }

# ------ Steps --------------------------------------------------------------------
function Check-Project {
    if (-not (Test-Path (Join-Path $PROJECT_DIR 'setup.py'))) { Write-Err "setup.py not found in $PROJECT_DIR"; return $false }
    Write-Ok "Project:  $PROJECT_DIR"
    Write-Ok "Version:  $(Get-Version)"
    $good = $true
    foreach ($f in 'README.md', 'LICENSE', 'MANIFEST.in', 'pyproject.toml') {
        if (Test-Path (Join-Path $PROJECT_DIR $f)) { Write-Ok "$f present" } else { Write-Err "$f missing"; $good = $false }
    }
    if (Test-DistReady) { Write-Ok 'Dashboard dist present' } else { Write-Warn "Dashboard dist missing - run 'Build dashboard'" }
    if ((Test-Path (Join-Path $PROJECT_DIR '.git')) -and (Test-Native 'git')) {
        $st = & git -C $PROJECT_DIR status --porcelain 2>$null
        if ($st) { Write-Warn 'Uncommitted changes in git tree' } else { Write-Ok 'Git tree clean' }
    }
    return $good
}

function Build-Dashboard {
    if (-not (Test-Path (Join-Path $DASH_DIR 'package.json'))) { Write-Err 'Dashboard package.json not found'; return $false }
    $mgr = $null
    if (Test-Native 'bun') { $mgr = 'bun' } elseif (Test-Native 'npm') { $mgr = 'npm' }
    if (-not $mgr) { Write-Err 'Neither bun nor npm found - install Node.js first'; return $false }
    Write-Info "Building dashboard with $mgr (fresh dist)..."
    $dist = Join-Path $DASH_DIR 'dist'
    if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }
    if ($mgr -eq 'bun') {
        $inst = if (Test-Path (Join-Path $DASH_DIR 'bun.lock')) { @('install', '--frozen-lockfile') } else { @('install') }
        $ok = (Invoke-Native 'bun' $inst $DASH_DIR) -and (Invoke-Native 'bun' @('run', 'build') $DASH_DIR)
    } else {
        $inst = if (Test-Path (Join-Path $DASH_DIR 'package-lock.json')) { 'ci' } else { 'install' }
        $ok = (Invoke-Native 'npm' @($inst) $DASH_DIR) -and (Invoke-Native 'npm' @('run', 'build') $DASH_DIR)
    }
    if (-not $ok -or -not (Test-DistReady)) { Write-Err 'Dashboard build failed (dist/index.html missing)'; return $false }
    Get-ChildItem $dist -Recurse -Filter '*.map' -File | ForEach-Object { Write-Warn "Removing source map $($_.Name)"; Remove-Item $_.FullName -Force }
    Write-Ok "dist built: $((Get-ChildItem $dist -Recurse -File).Count) files"
    return $true
}

function New-Stage {
    $stage = Join-Path ([System.IO.Path]::GetTempPath()) ("pypimgr_" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $stage | Out-Null
    Write-Info 'Staging clean copy of project...'
    # robocopy: /E recurse, /XD exclude dirs (by name or full path), /XF exclude files
    $xd = @('.git', 'build', 'node_modules', '__pycache__', '.venv', '.idea', '.vscode', '*.egg-info',
            (Join-Path $PROJECT_DIR 'dist'), (Join-Path $PROJECT_DIR 'scripts'),
            (Join-Path $DASH_DIR 'src'), (Join-Path $DASH_DIR 'backend\testing'))
    $xf = @('.env', '.env.*', '*.pem', '*.key', '.pypirc', '*.map', '*.pkl', '*.pt', '*.pth', '*.log', '*.pyc', '.DS_Store', 'TODO.md')
    $null = & robocopy $PROJECT_DIR $stage /E /NFL /NDL /NJH /NJS /NP /XD @xd /XF @xf
    if ($LASTEXITCODE -ge 8) { Write-Err "Staging failed (robocopy code $LASTEXITCODE)"; return $null }
    Write-Ok "Staged at $stage"
    return $stage
}

function Verify-Package {
    $files = @(Get-Artifacts)
    if ($files.Count -eq 0) { Write-Err 'No artifacts - build first'; return $false }
    if (-not (Ensure-Venv)) { return $false }
    Write-Info 'Inspecting artifacts for forbidden files / secrets...'
    if (-not (Invoke-Native $VENV_PY (@($VERIFY_PY) + $files))) { Write-Err 'Verification FAILED - do not upload'; return $false }
    if (-not (Invoke-Native $VENV_PY (@('-m', 'twine', 'check') + $files))) { Write-Err 'twine check failed'; return $false }
    Write-Ok 'Artifacts passed verification'
    return $true
}

function Build-Package {
    if (-not (Test-DistReady)) {
        Write-Warn 'Dashboard dist missing.'
        if (-not (Confirm-Yes 'Build it now?')) { return $false }
        if (-not (Build-Dashboard)) { return $false }
    }
    if (-not (Ensure-Venv)) { return $false }
    $stage = New-Stage
    if (-not $stage) { return $false }
    try {
        Write-Info 'Building sdist + wheel (dashboard build skipped; prebuilt dist is used)...'
        $env:MODIFIEDNEAT_SKIP_DASHBOARD_BUILD = '1'
        $out = Join-Path $stage '_out'
        $ok = Invoke-Native $VENV_PY @('-m', 'build', '--outdir', $out) $stage
        Remove-Item Env:\MODIFIEDNEAT_SKIP_DASHBOARD_BUILD -ErrorAction SilentlyContinue
        if (-not $ok) { Write-Err 'Build failed'; return $false }
        Get-ChildItem $ART_DIR -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '\.(whl|tar\.gz)$' } | Remove-Item -Force
        Copy-Item (Join-Path $out '*') $ART_DIR
        Write-Ok "Artifacts in $ART_DIR"
        @(Get-Artifacts) | ForEach-Object { Write-Host "        $(Split-Path $_ -Leaf)" }
    } finally {
        if (Test-Path $stage) { Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue }
    }
    return (Verify-Package)
}

function List-Contents {
    $files = @(Get-Artifacts)
    if ($files.Count -eq 0) { Write-Err 'No artifacts - build first'; return $false }
    if (-not (Ensure-Venv)) { return $false }
    foreach ($f in $files) {
        Write-Host ''; Write-Info (Split-Path $f -Leaf)
        if ($f -like '*.whl') { & $VENV_PY -m zipfile -l $f | Select-Object -First 400 }
        else { & tar -tzf $f | Select-Object -First 400 }
    }
    return $true
}

function Upload-Package([string]$Target) {
    $extra = @(); $label = 'PyPI (PRODUCTION)'
    if ($Target -eq 'test') { $label = 'TestPyPI'; $extra = @('--repository-url', $TESTPYPI_URL) }
    if (-not (Verify-Package)) { return $false }
    $ver = Get-Version
    Write-Warn "About to upload ModifiedNEAT $ver to $label. Published versions cannot be replaced."
    $typed = Read-Host "  Type the version ($ver) to confirm"
    if ($typed -ne $ver) { Write-Info 'Cancelled.'; return $true }
    $token = $env:PYPI_TOKEN
    $plain = $null
    if (-not $token) {
        $sec  = Read-Host '  API token (hidden, not stored)' -AsSecureString
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
        try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    } else { $plain = $token }
    if (-not $plain -or -not $plain.StartsWith('pypi-')) { Write-Err "Token must start with 'pypi-'"; return $false }
    $oldU = $env:TWINE_USERNAME; $oldP = $env:TWINE_PASSWORD
    try {
        $env:TWINE_USERNAME = '__token__'; $env:TWINE_PASSWORD = $plain
        $ok = Invoke-Native $VENV_PY (@('-m', 'twine', 'upload', '--non-interactive') + $extra + @(Get-Artifacts))
        if ($ok) { Write-Ok "Uploaded to $label" } else { Write-Err 'Upload failed' }
    } finally {
        $env:TWINE_USERNAME = $oldU; $env:TWINE_PASSWORD = $oldP
        if ($null -eq $oldU) { Remove-Item Env:\TWINE_USERNAME -ErrorAction SilentlyContinue }
        if ($null -eq $oldP) { Remove-Item Env:\TWINE_PASSWORD -ErrorAction SilentlyContinue }
        $plain = $null
    }
    return $true
}

function Bump-Version {
    Write-Info "Current version: $(Get-Version)"
    $new = Read-Host '  New version (blank = cancel)'
    if (-not $new) { return $true }
    if ($new -notmatch '^\d+(\.\d+)*([abrc]+\d+|\.post\d+|\.dev\d+)?$') { Write-Err 'Invalid version format'; return $false }
    $p = Join-Path $PROJECT_DIR 'setup.py'
    $raw = [System.IO.File]::ReadAllText($p)
    $raw = [regex]::Replace($raw, '(?m)^(\s*version=")[^"]+(")', "`${1}$new`${2}", 1)
    [System.IO.File]::WriteAllText($p, $raw, (New-Object System.Text.UTF8Encoding($false)))
    Write-Ok "setup.py version -> $(Get-Version)"
    return $true
}

function Clean-All {
    if (-not (Confirm-Yes 'Delete artifacts, build venv, build/ and *.egg-info?')) { return $true }
    foreach ($p in $VENV_DIR, $ART_DIR, (Join-Path $PROJECT_DIR 'build'), (Join-Path $PROJECT_DIR 'dist')) {
        if (Test-Path $p) { Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue }
    }
    Get-ChildItem $PROJECT_DIR -Directory -Filter '*.egg-info' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Ok 'Cleaned'
    return $true
}

function Full-Release {
    if (-not (Check-Project)) { return $false }
    if (-not (Build-Dashboard)) { return $false }
    if (-not (Build-Package)) { return $false }
    Write-Ok 'Ready. Use the upload options to publish.'
    return $true
}

# ------ Menu ---------------------------------------------------------------------
function Show-Menu {
    while ($true) {
        Clear-Host
        Write-Banner "ModifiedNEAT PyPI Manager  -  v$(Get-Version)"
        Write-Host '   1) Check project            6) Upload to TestPyPI'
        Write-Host '   2) Build dashboard (dist)   7) Upload to PyPI'
        Write-Host '   3) Build package (staged)   8) Bump version'
        Write-Host '   4) Verify artifacts         9) Clean'
        Write-Host '   5) List artifact contents   F) Full prepare (1+2+3)'
        Write-Host '                               Q) Quit'
        Write-Hr
        $c = (Read-Host '  Select').Trim().ToLower()
        Write-Host ''
        try {
            switch ($c) {
                '1' { [void](Check-Project) }
                '2' { [void](Build-Dashboard) }
                '3' { [void](Build-Package) }
                '4' { [void](Verify-Package) }
                '5' { [void](List-Contents) }
                '6' { [void](Upload-Package 'test') }
                '7' { [void](Upload-Package 'prod') }
                '8' { [void](Bump-Version) }
                '9' { [void](Clean-All) }
                'f' { [void](Full-Release) }
                'q' { return }
                default { Write-Warn 'Unknown option' }
            }
        } catch { Write-Err $_.Exception.Message }
        Wait-AnyKey
    }
}

if (-not (Test-Path (Join-Path $PROJECT_DIR 'setup.py'))) { Write-Err "setup.py not found in $PROJECT_DIR (set `$env:PROJECT_DIR)"; exit 1 }
Show-Menu
