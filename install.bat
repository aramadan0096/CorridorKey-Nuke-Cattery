@echo off
setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   CorridorKey for Nuke ^| Windows Installer
echo ============================================================
echo.

REM ── Step 1: Install uv via winget ────────────────────────────────────────
echo [1/2] Installing uv package manager via winget...
echo.
winget install --id astral-sh.uv -e --source winget
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   NOTE: winget returned a non-zero exit code.
    echo   This is expected if uv is already installed.
    echo   Continuing...
)

REM Refresh PATH for the current session so the freshly installed uv is found.
REM uv installs to %%LOCALAPPDATA%%\uv\bin on Windows.
set "PATH=%LOCALAPPDATA%\uv\bin;%PATH%"

REM Verify uv is available
uv --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ERROR: uv was not found after installation.
    echo   Please close this window, open a new terminal, and run install.bat again.
    echo   If the problem persists, install uv manually:
    echo     https://docs.astral.sh/uv/getting-started/installation/
    echo.
    pause
    exit /b 1
)

echo.
echo   uv is available.

REM ── Step 2: Install Python environment and all dependencies ──────────────
echo.
echo [2/2] Installing Python environment and dependencies (uv sync)...
echo.
cd /d "%~dp0"
uv sync
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ERROR: uv sync failed.
    echo   Check the output above for details.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Installation complete.
echo.
echo   Run start.bat to download the checkpoint and export the
echo   TorchScript model for Nuke.
echo ============================================================
echo.
pause
endlocal
