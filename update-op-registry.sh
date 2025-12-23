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

# Script to update framework import op registry configurations
# This script compiles and runs the OpRegistryUpdater utility

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

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
MAVEN_PROFILES=""

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
Update OP Registry Script
========================

Updates the available ops for import configuration files for framework import.
This script builds the project and runs the OpRegistryUpdater utility.

Usage: $0 [options]

Options:
  --framework <n>     Update specific framework only (tensorflow|onnx|all)
                        Default: all

  --validate-only       Only validate existing configurations without saving
                        Useful for testing and verification
                        Default: false

  --clean               Perform a clean build before running
                        Default: false

  --profiles <profiles> Additional Maven profiles to activate (comma-separated)
                        Example: --profiles cuda,testresources
                        Default: none

  --verbose             Enable verbose output
                        Default: false

  --help, -h            Show this help message

Examples:
  $0
      Update both TensorFlow and ONNX registries

  $0 --framework tensorflow
      Update only TensorFlow registry

  $0 --framework onnx --validate-only
      Validate ONNX registry without saving changes

  $0 --clean --profiles cuda
      Clean build with CUDA profile and update all registries

  $0 --framework tensorflow --verbose
      Update TensorFlow registry with verbose output

Requirements:
  - Maven 3.6+
  - Java 11+
  - Sufficient memory (recommend 8GB+ heap for large builds)

Note: This script will automatically build the required modules before running
      the registry updater. The first run may take longer due to compilation.
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
            --profiles)
                MAVEN_PROFILES="$2"
                shift 2
                ;;
            --profiles=*)
                MAVEN_PROFILES="${1#*=}"
                shift
                ;;
            --verbose)
                VERBOSE=true
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
        print_error "Not in the project root directory. Please run this script from the deeplearning4j root."
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Function to build the project
build_project() {
    print_info "Building required modules..."

    local maven_args=()
    
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        maven_args+=("clean")
    fi
    
    maven_args+=("compile")
    
    if [[ -n "$MAVEN_PROFILES" ]]; then
        maven_args+=("-P$MAVEN_PROFILES")
    fi
    
    maven_args+=("-DskipTests=true")
    
    if [[ "$VERBOSE" == "false" ]]; then
        maven_args+=("-q")
    fi

    # Build the core framework import modules
    print_info "Building samediff-import modules..."
    
    local modules=(
        "nd4j/samediff-import/samediff-import-api"
        "nd4j/samediff-import/samediff-import-tensorflow" 
        "nd4j/samediff-import/samediff-import-onnx"
        "platform-tests"
    )

    for module in "${modules[@]}"; do
        if [[ -d "$PROJECT_ROOT/$module" ]]; then
            print_info "Building module: $module"
            
            cd "$PROJECT_ROOT/$module"
            if [[ "$VERBOSE" == "true" ]]; then
                mvn "${maven_args[@]}"
            else
                mvn "${maven_args[@]}" > /dev/null 2>&1
            fi
            
            if [[ $? -ne 0 ]]; then
                print_error "Failed to build module: $module"
                exit 1
            fi
        else
            print_warning "Module directory not found: $module"
        fi
    done

    cd "$PROJECT_ROOT"
    print_success "Build completed successfully"
}

# Function to create and compile the OpRegistryUpdater if needed
setup_updater() {
    local updater_dir="$PROJECT_ROOT/platform-tests/src/main/kotlin/org/eclipse/deeplearning4j/frameworkimport/runner"
    local updater_file="$updater_dir/OpRegistryUpdater.kt"
    
    # Create directory if it doesn't exist
    mkdir -p "$updater_dir"
    
    # Check if the OpRegistryUpdater.kt file exists
    if [[ ! -f "$updater_file" ]]; then
        print_error "OpRegistryUpdater.kt not found at $updater_file"
        print_error "Please ensure the OpRegistryUpdater.kt file is placed in the correct location."
        exit 1
    fi
    
    print_info "OpRegistryUpdater found at $updater_file"
}

# Function to run the registry updater
run_updater() {
    print_info "Running OP Registry Updater..."
    print_info "Framework: $FRAMEWORK"
    print_info "Validate only: $VALIDATE_ONLY"
    
    cd "$PROJECT_ROOT"
    
    local maven_args=(
        "exec:java"
        "-Dexec.mainClass=org.eclipse.deeplearning4j.frameworkimport.runner.OpRegistryUpdater"
        "-pl" "platform-tests"
    )
    
    local exec_args="--framework=$FRAMEWORK"
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        exec_args="$exec_args --validate-only"
    fi
    
    maven_args+=("-Dexec.args=$exec_args")
    
    if [[ -n "$MAVEN_PROFILES" ]]; then
        maven_args+=("-P$MAVEN_PROFILES")
    fi
    
    if [[ "$VERBOSE" == "false" ]]; then
        maven_args+=("-q")
    fi

    print_info "Executing: mvn ${maven_args[*]}"
    
    if mvn "${maven_args[@]}"; then
        print_success "OP Registry Update completed successfully!"
    else
        print_error "OP Registry Update failed!"
        exit 1
    fi
}

# Function to show summary
show_summary() {
    print_info "Summary:"
    print_info "  Framework: $FRAMEWORK"
    print_info "  Validate only: $VALIDATE_ONLY"
    print_info "  Clean build: $CLEAN_BUILD"
    if [[ -n "$MAVEN_PROFILES" ]]; then
        print_info "  Maven profiles: $MAVEN_PROFILES"
    fi
    print_info "  Verbose: $VERBOSE"
    echo
}

# Main execution
main() {
    print_info "Starting OP Registry Update Script..."
    
    parse_arguments "$@"
    show_summary
    check_prerequisites
    setup_updater
    build_project
    run_updater
    
    print_success "Script execution completed!"
}

# Run main function with all arguments
main "$@"