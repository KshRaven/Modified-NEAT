@echo off
REM ======================================================================
REM pypimgr.bat - ModifiedNEAT PyPI Build & Upload Manager (Windows)
REM Requirements: Windows 10+, Python 3.11+ (administrator rights NOT required)
REM   Runs on PowerShell 7+ (pwsh) if available, otherwise falls back to
REM   the built-in Windows PowerShell 5.1 - pypimgr.ps1 works on either.
REM Usage: pypimgr.bat
REM ======================================================================

setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM Resolve script directory
set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%pypimgr.ps1"
set "PS_EXE="

REM Check pypimgr.ps1 exists
if not exist "%PS_SCRIPT%" (
    echo.
    echo   [ERROR]  pypimgr.ps1 not found
    echo            Expected: %PS_SCRIPT%
    echo.
    pause
    exit /b 1
)

REM ----------------------------------------------------------------
REM Detect PowerShell 7+ (pwsh) - preferred, but not required
REM ----------------------------------------------------------------
where pwsh >nul 2>&1
if not errorlevel 1 (
    pwsh -NoProfile -Command "exit 0" >nul 2>&1
    if not errorlevel 1 set "PS_EXE=pwsh"
)

if defined PS_EXE goto :launch

REM ----------------------------------------------------------------
REM pwsh not available: offer to install it, or fall back to the
REM built-in Windows PowerShell 5.1 (fully supported by this script)
REM ----------------------------------------------------------------
echo.
echo   [!]  PowerShell 7+ (pwsh) was not found on this system.
echo        pypimgr works fine on the built-in Windows PowerShell too,
echo        PowerShell 7 just gives a slightly smoother terminal experience.
echo.

where winget >nul 2>&1
if errorlevel 1 (
    set "WINGET_OK=0"
) else (
    set "WINGET_OK=1"
)

if "%WINGET_OK%"=="1" (
    choice /C YNC /N /M "  Install PowerShell 7 now [Y], use Windows PowerShell instead [N], or cancel [C]: "
    if errorlevel 3 goto :cancelled
    if errorlevel 2 goto :use_winps
    if errorlevel 1 goto :install_pwsh
) else (
    echo   [i]  winget was not found, so automatic install isn't available here.
    echo        You can install PowerShell 7 manually later from:
    echo        https://learn.microsoft.com/en-us/powershell/scripting/install
    echo.
    choice /C NC /N /M "  Use Windows PowerShell instead [N], or cancel [C]: "
    if errorlevel 2 goto :cancelled
    if errorlevel 1 goto :use_winps
)

:install_pwsh
echo.
echo   [*]  Installing PowerShell 7 via winget - this may take a minute...
echo.
winget install --id Microsoft.PowerShell -e --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo   [ERROR]  winget install failed or was cancelled.
    echo            Falling back to Windows PowerShell for this run.
    goto :use_winps
)

REM winget usually registers pwsh on PATH, but the *current* window
REM won't see a PATH update until it's reopened. Check both ways.
where pwsh >nul 2>&1
if not errorlevel 1 (
    set "PS_EXE=pwsh"
    echo.
    echo   [OK]  PowerShell 7 installed successfully.
    goto :launch
)
if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" (
    set "PS_EXE=%ProgramFiles%\PowerShell\7\pwsh.exe"
    echo.
    echo   [OK]  PowerShell 7 installed successfully.
    goto :launch
)

echo.
echo   [!]  PowerShell 7 was installed, but this window can't see it yet.
echo        Using Windows PowerShell for this run - open a new terminal
echo        next time to pick up PowerShell 7 automatically.
goto :use_winps

:use_winps
where powershell >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [ERROR]  No usable PowerShell found ^(neither pwsh nor powershell.exe^).
    echo            Please install PowerShell:
    echo            https://learn.microsoft.com/en-us/powershell/scripting/install
    echo.
    pause
    exit /b 1
)
set "PS_EXE=powershell"
echo.
goto :launch

:cancelled
echo.
echo   Cancelled. Exiting.
echo.
exit /b 1

:launch
REM Launch pypimgr.ps1 with whichever PowerShell we resolved to
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%"

REM Propagate exit code
set EXIT_CODE=%errorlevel%
if !EXIT_CODE! neq 0 (
    if !EXIT_CODE! neq 130 (
        echo.
        echo   [ERROR]  PowerShell script exited with code !EXIT_CODE!
        echo.
        pause
    )
)
exit /b !EXIT_CODE!