@echo off
dotnet run --project "%~dp0lessons\miniGPT\MiniGPT.csproj" -- %*
exit /b %errorlevel%
