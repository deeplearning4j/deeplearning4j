#!/bin/bash

#
# ******************************************************************************
# *
# *
# * This program and the accompanying materials are made available under the
# * terms of the Apache License, Version 2.0 which is available at
# * https://www.apache.org/licenses/LICENSE-2.0.
# *
# *  See the NOTICE file distributed with this work for additional
# *  information regarding copyright ownership.
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# * License for the specific language governing permissions and limitations
# * under the License.
# *
# * SPDX-License-Identifier: Apache-2.0
# *****************************************************************************
#

# Standalone script to update framework import op registry configurations
# This script compiles and runs the standalone OpRegistryUpdater utility

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
FRAMEWORK="all"
VALIDATE_ONLY=false
CLEAN_BUILD=false
VERBOSE=false
USE_EXEC_JAR=false
DEBUG=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
Standalone OP Registry Update Script
====================================

Updates the available ops for import configuration files for framework import.
This standalone script builds and runs the OpRegistryUpdater utility without
requiring the platform-tests module.

Usage: $0 [options]

Options:
  --framework <n>       Update specific framework only (tensorflow|onnx|all)
                          Default: all

  --validate-only         Only validate existing configurations without saving
                          Useful for testing and verification
                          Default: false

  --clean                 Perform a clean build before running
                          Default: false

  --verbose               Enable verbose output
                          Default: false

  --debug                 Enable debug output for PreImportHook discovery
                          Useful for troubleshooting missing hooks
                          Default: false

  --use-jar               Use executable JAR instead of maven exec:java
                          Default: false (use maven exec:java)

  --help, -h              Show this help message

Examples:
  $0
      Update both TensorFlow and ONNX registries

  $0 --framework tensorflow
      Update only TensorFlow registry

  $0 --framework onnx --validate-only
      Validate ONNX registry without saving changes

  $0 --framework onnx --debug
      Update ONNX registry with debug output to check PreImportHook discovery

  $0 --clean --use-jar
      Clean build and run via executable JAR

  $0 --framework tensorflow --verbose
      Update TensorFlow registry with verbose output

Requirements:
  - Maven 3.6+
  - Java 11+
  - Sufficient memory (recommend 8GB+ heap for large builds)

Note: This standalone script is independent of the platform-tests module and
      will automatically build only the required dependencies.
EOF
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --framework)
                FRAMEWORK="$2"
                if [[ ! "$FRAMEWORK" =~ ^(tensorflow|onnx|all)$ ]]; then
                    print_error "Invalid framework: $FRAMEWORK. Must be tensorflow, onnx, or all"
                    exit 1
                fi
                shift 2
                ;;
            --framework=*)
                FRAMEWORK="${1#*=}"
                if [[ ! "$FRAMEWORK" =~ ^(tensorflow|onnx|all)$ ]]; then
                    print_error "Invalid framework: $FRAMEWORK. Must be tensorflow, onnx, or all"
                    exit 1
                fi
                shift
                ;;
            --validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            --use-jar)
                USE_EXEC_JAR=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check Java
    if ! command -v java &> /dev/null; then
        print_error "Java is not installed or not in PATH"
        exit 1
    fi

    local java_version
    java_version=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1-2)
    print_info "Found Java version: $java_version"

    # Check Maven
    if ! command -v mvn &> /dev/null; then
        print_error "Maven is not installed or not in PATH"
        exit 1
    fi

    local maven_version
    maven_version=$(mvn -version | head -n 1 | cut -d' ' -f3)
    print_info "Found Maven version: $maven_version"

    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/pom.xml" ]]; then
        print_error "Not in the project root directory. Please run this script from the contrib/op-registry-updater directory."
        exit 1
    fi

    # Check if the op-registry-updater module exists
    if [[ ! -f "$SCRIPT_DIR/pom.xml" ]]; then
        print_error "op-registry-updater module not found. Please ensure you're running from contrib/op-registry-updater/"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Function to build dependencies
build_dependencies() {
    print_info "Building required dependencies..."

    local maven_args=()
    
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        maven_args+=("clean")
    fi
    
    maven_args+=("compile")
    maven_args+=("-DskipTests=true")
    
    if [[ "$VERBOSE" == "false" ]]; then
        maven_args+=("-q")
    fi

    # Build the core framework import modules from project root
    cd "$PROJECT_ROOT"
    
    local modules=(
        "nd4j/samediff-import/samediff-import-api"
        "nd4j/samediff-import/samediff-import-tensorflow" 
        "nd4j/samediff-import/samediff-import-onnx"
    )

    for module in "${modules[@]}"; do
        if [[ -d "$PROJECT_ROOT/$module" ]]; then
            print_info "Building dependency: $module"
            
            if [[ "$VERBOSE" == "true" ]]; then
                mvn "${maven_args[@]}" -pl "$module"
            else
                mvn "${maven_args[@]}" -pl "$module" > /dev/null 2>&1
            fi
            
            if [[ $? -ne 0 ]]; then
                print_error "Failed to build dependency: $module"
                exit 1
            fi
        else
            print_warning "Dependency directory not found: $module"
        fi
    done

    print_success "Dependencies built successfully"
}

# Function to build the standalone module
build_standalone_module() {
    print_info "Building standalone op-registry-updater module..."

    cd "$SCRIPT_DIR"
    
    local maven_args=()
    
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        maven_args+=("clean")
    fi
    
    maven_args+=("compile")
    
    if [[ "$USE_EXEC_JAR" == "true" ]]; then
        maven_args+=("package")
    fi
    
    if [[ "$VERBOSE" == "false" ]]; then
        maven_args+=("-q")
    fi

    print_info "Executing: mvn ${maven_args[*]}"
    
    if [[ "$VERBOSE" == "true" ]]; then
        mvn "${maven_args[@]}"
    else
        mvn "${maven_args[@]}" > /dev/null 2>&1
    fi
    
    if [[ $? -ne 0 ]]; then
        print_error "Failed to build standalone module"
        exit 1
    fi

    print_success "Standalone module built successfully"
}

# Function to run the registry updater
run_updater() {
    print_info "Running Standalone OP Registry Updater..."
    print_info "Framework: $FRAMEWORK"
    print_info "Validate only: $VALIDATE_ONLY"
    print_info "Debug mode: $DEBUG"
    
    cd "$SCRIPT_DIR"
    
    local exec_args="--framework=$FRAMEWORK"
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        exec_args="$exec_args --validate-only"
    fi
    if [[ "$DEBUG" == "true" ]]; then
        exec_args="$exec_args --debug"
    fi

    if [[ "$USE_EXEC_JAR" == "true" ]]; then
        # Run using the shaded JAR
        local jar_file="target/op-registry-updater-1.0.0-SNAPSHOT.jar"
        
        if [[ ! -f "$jar_file" ]]; then
            print_error "Executable JAR not found: $jar_file"
            print_error "Please run with --clean to rebuild the JAR"
            exit 1
        fi
        
        print_info "Executing JAR: java -jar $jar_file $exec_args"
        java -jar "$jar_file" $exec_args
    else
        # Run using maven exec:java
        local maven_args=(
            "exec:java"
            "-Dexec.args=$exec_args"
        )
        
        if [[ "$VERBOSE" == "false" ]]; then
            maven_args+=("-q")
        fi

        print_info "Executing: mvn ${maven_args[*]}"
        mvn "${maven_args[@]}"
    fi
    
    if [[ $? -eq 0 ]]; then
        print_success "OP Registry Update completed successfully!"
        
        if [[ "$DEBUG" == "true" ]]; then
            print_info ""
            print_info "Debug Summary:"
            print_info "- If EmbedLayerNormalization was not found, the issue is likely:"
            print_info "  1. The implementations directory is not in the classpath during execution"
            print_info "  2. The ClassGraph scanner is not finding the @PreHookRule annotations"
            print_info "  3. The class needs to be compiled and available when the registry runs"
            print_info ""
            print_info "To fix this, ensure that samediff-import-onnx is properly built and"
            print_info "the implementations package is included in the classpath."
        fi
    else
        print_error "OP Registry Update failed!"
        exit 1
    fi
}

# Function to show summary
show_summary() {
    print_info "Configuration Summary:"
    print_info "  Framework: $FRAMEWORK"
    print_info "  Validate only: $VALIDATE_ONLY"
    print_info "  Clean build: $CLEAN_BUILD"
    print_info "  Verbose: $VERBOSE"
    print_info "  Debug: $DEBUG"
    print_info "  Use executable JAR: $USE_EXEC_JAR"
    print_info "  Script directory: $SCRIPT_DIR"
    print_info "  Project root: $PROJECT_ROOT"
    echo
}

# Main execution
main() {
    print_info "Starting Standalone OP Registry Update Script..."
    echo
    
    parse_arguments "$@"
    show_summary
    check_prerequisites
    build_dependencies
    build_standalone_module
    run_updater
    
    echo
    print_success "Script execution completed successfully!"
    print_info "This standalone tool has replaced the need for platform-tests module dependency."
}

# Run main function with all arguments
main "$@"
