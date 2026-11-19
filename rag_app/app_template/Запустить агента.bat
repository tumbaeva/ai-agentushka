@echo off
chcp 65001 >nul
title {{AGENT_NAME}}

set "LOG=%~dp0start.log"
>"%LOG%" echo [%date% %time%] start
>>"%LOG%" echo initial-dir=%CD%
>>"%LOG%" echo script=%~f0

echo Запуск {{AGENT_NAME}}...

cd /d "%~dp0"
>>"%LOG%" echo cd-ok=%CD%

rem ---- application files must be present next to this script ----
if exist "run.py" goto have_files
>>"%LOG%" echo ERROR no-run.py
echo [ошибка] Рядом нет файлов приложения: run.py, rag_app, requirements.txt
echo Распакуйте архив целиком и запустите файл из распакованной папки.
echo Путь запуска сейчас: %~dp0
goto end
:have_files

rem ---- a ';' in the folder path breaks venv creation, so use a safe location ----
set "VENV=.venv"
echo "%~dp0." | find ";" >nul
if not errorlevel 1 set "VENV=%LOCALAPPDATA%\{{AGENT_FILE}}_venv"
>>"%LOG%" echo venv=%VENV%

rem ---- find a working Python ----
set "PY="
python --version >nul 2>&1 && set "PY=python"
if not defined PY py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY goto no_python
>>"%LOG%" echo python=%PY%
goto have_python

:no_python
>>"%LOG%" echo ERROR no-python
echo [ошибка] Python не найден.
echo Установите Python 3.10+ с https://www.python.org/downloads/windows/
echo и включите галочку "Add python.exe to PATH" при установке.
goto end

:have_python
rem ---- create a desktop shortcut once (points to this launcher) ----
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $p=Join-Path $d '{{AGENT_FILE}}.lnk'; if(-not (Test-Path $p)){ $s=New-Object -ComObject WScript.Shell; $l=$s.CreateShortcut($p); $l.TargetPath=Join-Path $env:SystemRoot 'System32\cmd.exe'; $l.Arguments='/c \"\"%~f0\"\"'; $l.WorkingDirectory='%~dp0'; $l.IconLocation='%SystemRoot%\System32\shell32.dll,13'; $l.Save() }" >nul 2>&1

rem ---- first run: create venv and install dependencies ----
if exist "%VENV%\Scripts\python.exe" goto run_app
echo Первый запуск: создаю окружение и устанавливаю зависимости, это может занять несколько минут...
echo Для этого нужно примерно 3-4 ГБ свободного места на диске %~d0.
>>"%LOG%" echo creating-venv
%PY% -m venv "%VENV%"
if errorlevel 1 goto venv_failed
>>"%LOG%" echo installing
rem keep pip from hoarding a download cache and use the app drive for temp
set "PIP_NO_CACHE_DIR=1"
set "TEMP=%VENV%\_tmp"
set "TMP=%VENV%\_tmp"
if not exist "%VENV%\_tmp" md "%VENV%\_tmp"
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto pip_failed
"%VENV%\Scripts\python.exe" -m pip cache purge >nul 2>&1
goto run_app

:venv_failed
>>"%LOG%" echo ERROR venv-failed
echo [ошибка] Не удалось создать виртуальное окружение.
goto end

:pip_failed
>>"%LOG%" echo ERROR pip-failed
echo.
echo [ошибка] Не удалось установить зависимости — чаще всего из-за нехватки места на диске.
echo Освободите место на диске %~d0 ИЛИ распакуйте архив на диск посвободнее
echo и запустите "Запустить агента.bat" заново уже оттуда.
echo Ошибка pip напечатана выше.
goto end

:run_app
>>"%LOG%" echo starting-app
echo Запускаю {{AGENT_NAME}}... Браузер откроется автоматически.
echo Если браузер не открылся - скопируйте адрес из строки ниже.
"%VENV%\Scripts\python.exe" run.py
if errorlevel 1 goto app_failed
>>"%LOG%" echo app-closed
goto :eof

:app_failed
>>"%LOG%" echo ERROR app-exited
echo.
echo [ошибка] Агент завершился с ошибкой. Подробности напечатаны выше.
goto end

:end
echo.
pause
exit /b 1
