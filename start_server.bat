@echo off
title DeepShield-X Web Server
echo Starting DeepShield-X Server on http://localhost:8080 ...
powershell -ExecutionPolicy Bypass -File "%~dp0serve.ps1"
pause

