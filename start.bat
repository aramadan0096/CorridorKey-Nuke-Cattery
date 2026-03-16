@echo off
setlocal

REM Refresh PATH so uv is found if installed in the same session as install.bat.
set "PATH=%LOCALAPPDATA%\uv\bin;%PATH%"

echo.
echo ============================================================
echo   CorridorKey for Nuke ^| Bootstrap
echo   Downloads checkpoint + exports TorchScript model
echo ============================================================
echo.

cd /d "%~dp0"
uv run python bootstrap.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ERROR: bootstrap.py failed (exit code %ERRORLEVEL%).
    echo   Check the output above for details.
    echo.
    pause
    exit /b 1
)

pause
endlocal
