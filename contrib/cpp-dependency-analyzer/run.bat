@echo off
setlocal enabledelayedexpansion
REM C++ Dependency Analyzer Runner Script for Windows
REM Usage: run.bat [options] <root-directory>

REM Build the project if JAR doesn't exist
set JAR_FILE=target\cpp-dependency-analyzer-1.0.0-SNAPSHOT.jar

if not exist "%JAR_FILE%" (
    echo Building project...
    call mvn clean package -q
    if errorlevel 1 (
        echo Build failed!
        exit /b 1
    )
)

REM Run the analyzer with properly quoted arguments
REM Build argument list with proper quoting to prevent command injection
set "ARGS="
:parse_args
if "%~1"=="" goto run_analyzer
set "ARGS=!ARGS! "%~1""
shift
goto parse_args

:run_analyzer
if defined ARGS (
    java -jar "%JAR_FILE%" %ARGS%
) else (
    java -jar "%JAR_FILE%"
)

endlocal
