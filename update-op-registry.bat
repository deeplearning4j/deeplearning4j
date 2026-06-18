@echo off
REM ******************************************************************************
REM *
REM *
REM * This program and the accompanying materials are made available under the
REM * terms of the Apache License, Version 2.0 which is available at
REM * https://www.apache.org/licenses/LICENSE-2.0.
REM *
REM *  See the NOTICE file distributed with this work for additional
REM *  information regarding copyright ownership.
REM * Unless required by applicable law or agreed to in writing, software
REM * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
REM * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
REM * License for the specific language governing permissions and limitations
REM * under the License.
REM *
REM * SPDX-License-Identifier: Apache-2.0
REM *****************************************************************************

REM Windows batch script to update framework import op registry configurations

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%"

REM Default values
set "FRAMEWORK=all"
set "VALIDATE_ONLY=false"
set "CLEAN_BUILD=false"
set "VERBOSE=false"
set "MAVEN_PROFILES="

REM Colors (limited support in Windows)
set "INFO_PREFIX=[INFO]"
set "ERROR_PREFIX=[ERROR]"
set "SUCCESS_PREFIX=[SUCCESS]"
set "WARNING_PREFIX=[WARNING]"

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--help" goto show_help
if "%~1"=="-h" goto show_help
if "%~1"=="--validate-only" (
    set "VALIDATE_ONLY=true"
    shift
    goto parse_args
)
if "%~1"=="--clean" (
    set "CLEAN_BUILD=true"
    shift
    goto parse_args
)
if "%~1"=="--verbose" (
    set "VERBOSE=true"
    shift
    goto parse_args
)
if "%~1" NEQ "" (
    echo %ERROR_PREFIX% Unknown option: %~1
    goto show_help
)
shift
goto parse_args

:args_done

REM Check for framework parameter
for %%i in (%*) do (
    if "%%i" == "--framework=tensorflow" set "FRAMEWORK=tensorflow"
    if "%%i" == "--framework=onnx" set "FRAMEWORK=onnx"
    if "%%i" == "--framework=all" set "FRAMEWORK=all"
)

REM Check for profiles parameter
for %%i in (%*) do (
    echo %%i | findstr /C:"--profiles=" >nul
    if !errorlevel! == 0 (
        set "MAVEN_PROFILES=%%i"
        set "MAVEN_PROFILES=!MAVEN_PROFILES:--profiles==!"
    )
)

echo %INFO_PREFIX% Starting OP Registry Update Script...
echo %INFO_PREFIX% Framework: %FRAMEWORK%
echo %INFO_PREFIX% Validate only: %VALIDATE_ONLY%
echo %INFO_PREFIX% Clean build: %CLEAN_BUILD%
echo %INFO_PREFIX% Verbose: %VERBOSE%
if not "%MAVEN_PROFILES%"=="" echo %INFO_PREFIX% Maven profiles: %MAVEN_PROFILES%
echo.

REM Check prerequisites
echo %INFO_PREFIX% Checking prerequisites...

REM Check Java
java -version >nul 2>&1
if errorlevel 1 (
    echo %ERROR_PREFIX% Java is not installed or not in PATH
    exit /b 1
)
echo %INFO_PREFIX% Java found

REM Check Maven
mvn -version >nul 2>&1
if errorlevel 1 (
    echo %ERROR_PREFIX% Maven is not installed or not in PATH
    exit /b 1
)
echo %INFO_PREFIX% Maven found

REM Check if we're in the right directory
if not exist "%PROJECT_ROOT%\pom.xml" (
    echo %ERROR_PREFIX% Not in the project root directory. Please run this script from the deeplearning4j root.
    exit /b 1
)

echo %SUCCESS_PREFIX% Prerequisites check passed

REM Build required modules
echo %INFO_PREFIX% Building required modules...

set "MVN_ARGS=compile -DskipTests=true"

if "%CLEAN_BUILD%"=="true" (
    set "MVN_ARGS=clean %MVN_ARGS%"
)

if not "%MAVEN_PROFILES%"=="" (
    set "MVN_ARGS=%MVN_ARGS% -P%MAVEN_PROFILES%"
)

if "%VERBOSE%"=="false" (
    set "MVN_ARGS=%MVN_ARGS% -q"
)

REM Build each required module
set "MODULES=nd4j\samediff-import\samediff-import-api nd4j\samediff-import\samediff-import-tensorflow nd4j\samediff-import\samediff-import-onnx platform-tests"

for %%m in (%MODULES%) do (
    echo %INFO_PREFIX% Building module: %%m
    cd /d "%PROJECT_ROOT%\%%m"
    if "%VERBOSE%"=="true" (
        mvn %MVN_ARGS%
    ) else (
        mvn %MVN_ARGS% >nul 2>&1
    )
    if errorlevel 1 (
        echo %ERROR_PREFIX% Failed to build module: %%m
        exit /b 1
    )
    cd /d "%PROJECT_ROOT%"
)

echo %SUCCESS_PREFIX% Build completed successfully

REM Check if OpRegistryUpdater exists
set "UPDATER_FILE=%PROJECT_ROOT%\platform-tests\src\main\kotlin\org\eclipse\deeplearning4j\frameworkimport\runner\OpRegistryUpdater.kt"
if not exist "%UPDATER_FILE%" (
    echo %ERROR_PREFIX% OpRegistryUpdater.kt not found at %UPDATER_FILE%
    echo %ERROR_PREFIX% Please ensure the OpRegistryUpdater.kt file is placed in the correct location.
    exit /b 1
)

echo %INFO_PREFIX% OpRegistryUpdater found

REM Run the registry updater
echo %INFO_PREFIX% Running OP Registry Updater...

cd /d "%PROJECT_ROOT%"

set "EXEC_ARGS=--framework=%FRAMEWORK%"
if "%VALIDATE_ONLY%"=="true" (
    set "EXEC_ARGS=%EXEC_ARGS% --validate-only"
)

set "MVN_EXEC_ARGS=exec:java -Dexec.mainClass=org.eclipse.deeplearning4j.frameworkimport.runner.OpRegistryUpdater -pl platform-tests -Dexec.args=\"%EXEC_ARGS%\""

if not "%MAVEN_PROFILES%"=="" (
    set "MVN_EXEC_ARGS=%MVN_EXEC_ARGS% -P%MAVEN_PROFILES%"
)

if "%VERBOSE%"=="false" (
    set "MVN_EXEC_ARGS=%MVN_EXEC_ARGS% -q"
)

echo %INFO_PREFIX% Executing: mvn %MVN_EXEC_ARGS%

mvn %MVN_EXEC_ARGS%
if errorlevel 1 (
    echo %ERROR_PREFIX% OP Registry Update failed!
    exit /b 1
)

echo %SUCCESS_PREFIX% OP Registry Update completed successfully!
echo %SUCCESS_PREFIX% Script execution completed!
goto end

:show_help
echo Update OP Registry Script - Windows Version
echo =========================================
echo.
echo Updates the available ops for import configuration files for framework import.
echo This script builds the project and runs the OpRegistryUpdater utility.
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --framework=^<n^>     Update specific framework only (tensorflow^|onnx^|all)
echo                        Default: all
echo.
echo   --validate-only      Only validate existing configurations without saving
echo                        Default: false
echo.
echo   --clean              Perform a clean build before running
echo                        Default: false
echo.
echo   --profiles=^<profiles^> Additional Maven profiles (comma-separated)
echo                        Example: --profiles=cuda,testresources
echo                        Default: none
echo.
echo   --verbose            Enable verbose output
echo                        Default: false
echo.
echo   --help, -h           Show this help message
echo.
echo Examples:
echo   %~nx0
echo       Update both TensorFlow and ONNX registries
echo.
echo   %~nx0 --framework=tensorflow
echo       Update only TensorFlow registry
echo.
echo   %~nx0 --framework=onnx --validate-only
echo       Validate ONNX registry without saving changes
echo.
echo   %~nx0 --clean --profiles=cuda
echo       Clean build with CUDA profile and update all registries
echo.
echo Requirements:
echo   - Maven 3.6+
echo   - Java 11+
echo   - Sufficient memory (recommend 8GB+ heap for large builds)
echo.
goto end

:end
endlocal