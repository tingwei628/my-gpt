@echo off
dotnet run --project "%~dp0lessons\my-gpt\my-gpt.csproj" -- %*
exit /b %errorlevel%
