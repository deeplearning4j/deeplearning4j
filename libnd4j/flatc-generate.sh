#!/bin/bash
#
# /* ******************************************************************************
#  *
#  *
#  * This program and the accompanying materials are made available under the
#  * terms of the Apache License, Version 2.0 which is available at
#  * https://www.apache.org/licenses/LICENSE-2.0.
#  *
#  *  See the NOTICE file distributed with this work for additional
#  *  information regarding copyright ownership.
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#

# Exit on error
set -e

# Enable more verbose error reporting
set -o pipefail

if [ -z "$FLATC_PATH" ]; then
    echo "❌ FLATC_PATH not set"
    exit 1
fi

echo "🔧 Using flatc compiler from: $FLATC_PATH"
if [ ! -f "$FLATC_PATH" ]; then
    echo "❌ Error: flatc not found at $FLATC_PATH"
    exit 1
fi

if [ ! -x "$FLATC_PATH" ]; then
    echo "❌ Error: $FLATC_PATH exists but is not executable"
    exit 1
fi

# Function to safely copy directory contents without self-copying
safe_copy_directory() {
    local source_dir="$1"
    local dest_dir="$2"

    # Get absolute paths to avoid issues with relative paths
    source_dir=$(realpath "$source_dir" 2>/dev/null || echo "$source_dir")
    dest_dir=$(realpath "$dest_dir" 2>/dev/null || echo "$dest_dir")

    if [ ! -d "$source_dir" ]; then
        echo "⚠️  Warning: Source directory does not exist: $source_dir"
        return 0
    fi

    # Create destination directory
    mkdir -p "$dest_dir"

    # Check if source and destination are the same
    if [ "$source_dir" = "$dest_dir" ]; then
        echo "ℹ️  Source and destination are identical, skipping copy: $source_dir"
        return 0
    fi

    # Check if destination is a subdirectory of source to avoid infinite recursion
    case "$dest_dir" in
        "$source_dir"/*)
            echo "⚠️  Warning: Destination is inside source directory, this could cause issues"
            echo "   Source: $source_dir"
            echo "   Dest: $dest_dir"
            ;;
    esac

    echo "📋 Copying contents from $source_dir to $dest_dir"

    # Use rsync if available for better handling, otherwise fall back to cp
    if command -v rsync >/dev/null 2>&1; then
        rsync -av --exclude="$dest_dir" "$source_dir"/ "$dest_dir"/
    else
        # Use find to avoid copying destination into itself
        find "$source_dir" -mindepth 1 -maxdepth 1 -not -path "$dest_dir" -exec cp -r {} "$dest_dir"/ \;
    fi

    echo "✅ Copy completed successfully"
}

# Find flatbuffers directory in any subdirectory
echo "🔍 Searching for flatbuffers directory..."
FLATBUFFERS_DIR=$(find . -name "flatbuffers" -type d | head -1)

if [ -z "$FLATBUFFERS_DIR" ]; then
    echo "⚠️  Warning: flatbuffers directory not found in any subdirectory"
    echo "   This might be expected if headers are already in place"
    echo "   Continuing with flatbuffer generation..."
else
    echo "✅ Found flatbuffers directory at: $FLATBUFFERS_DIR"

    # Create flatbuffers directory under ./include
    mkdir -p ./include/flatbuffers

    # Safely copy everything from the found flatbuffers directory to ./include/flatbuffers/
    safe_copy_directory "$FLATBUFFERS_DIR" "./include/flatbuffers"

    echo "📋 Verifying copied files:"
    ls -la ./include/flatbuffers/ || echo "⚠️  Directory listing failed"
fi

# Ensure output directories exist
mkdir -p ./include/graph/generated

# Check if required .fbs files exist before proceeding
FBS_FILES=(
    "./include/graph/scheme/node.fbs"
    "./include/graph/scheme/graph.fbs"
    "./include/graph/scheme/result.fbs"
    "./include/graph/scheme/request.fbs"
    "./include/graph/scheme/config.fbs"
    "./include/graph/scheme/array.fbs"
    "./include/graph/scheme/utils.fbs"
    "./include/graph/scheme/variable.fbs"
    "./include/graph/scheme/properties.fbs"
    "./include/graph/scheme/sequence.fbs"
)

UI_FBS_FILES=(
    "./include/graph/scheme/uigraphstatic.fbs"
    "./include/graph/scheme/uigraphevents.fbs"
)

echo "🔍 Checking for required .fbs files..."
missing_files=0
for fbs_file in "${FBS_FILES[@]}"; do
    if [ ! -f "$fbs_file" ]; then
        echo "❌ Missing required file: $fbs_file"
        missing_files=$((missing_files + 1))
    else
        echo "✅ Found: $fbs_file"
    fi
done

for fbs_file in "${UI_FBS_FILES[@]}"; do
    if [ ! -f "$fbs_file" ]; then
        echo "⚠️  Optional UI file missing: $fbs_file"
    else
        echo "✅ Found: $fbs_file"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "❌ Error: $missing_files required .fbs files are missing"
    exit 1
fi

echo "🚀 Starting flatbuffer generation..."

# Generate flatbuffer files using built flatc with correct package (first command)
echo "🔧 Generating flatbuffers with full options (Java, Binary, Text, C++)..."
if ! "$FLATC_PATH" -o ./include/graph/generated -I ./include/graph/scheme -j -b -t -c \
    --java-package-prefix org.nd4j \
    "${FBS_FILES[@]}"; then
    echo "❌ Error: First flatc command failed"
    exit 1
fi

# Generate flatbuffer files (second command - Java and Binary only)
echo "🔧 Generating flatbuffers (Java and Binary only)..."
if ! "$FLATC_PATH" -o ./include/graph/generated -I ./include/graph/scheme -j -b \
    --java-package-prefix org.nd4j \
    "${FBS_FILES[@]}"; then
    echo "❌ Error: Second flatc command failed"
    exit 1
fi

# Generate gRPC flatbuffers if UI files exist
ui_files_exist=true
for fbs_file in "${UI_FBS_FILES[@]}"; do
    if [ ! -f "$fbs_file" ]; then
        ui_files_exist=false
        break
    fi
done

if [ "$ui_files_exist" = true ]; then
    echo "🔧 Generating gRPC flatbuffers..."
    if ! "$FLATC_PATH" -o ./include/graph/generated -I ./include/graph/scheme -j -b --grpc \
        --java-package-prefix org.nd4j \
        "${UI_FBS_FILES[@]}"; then
        echo "❌ Error: gRPC flatc command failed"
        exit 1
    fi
else
    echo "⚠️  Skipping gRPC generation - UI files not found"
fi

# Generate TypeScript files if target directory exists
TS_OUTPUT_DIR="../nd4j/nd4j-web/nd4j-webjar/src/main/typescript"
if [ -d "$(dirname "$TS_OUTPUT_DIR")" ]; then
    echo "🔧 Generating TypeScript files..."
    mkdir -p "$TS_OUTPUT_DIR"

    # Combine all FBS files for TypeScript generation
    ALL_TS_FILES=("${FBS_FILES[@]}")
    if [ "$ui_files_exist" = true ]; then
        ALL_TS_FILES+=("${UI_FBS_FILES[@]}")
    fi

    if ! "$FLATC_PATH" -o "$TS_OUTPUT_DIR" -I ./include/graph/scheme --ts \
        --ts-flat-files \
        "${ALL_TS_FILES[@]}"; then
        echo "❌ Error: TypeScript flatc command failed"
        exit 1
    fi
    echo "✅ TypeScript files generated successfully"
else
    echo "ℹ️  Skipping TypeScript generation - target directory doesn't exist: $TS_OUTPUT_DIR"
fi

echo "✅ Flatbuffer sources generated successfully"

# Verify that some expected output files were created
echo "🔍 Verifying generated files..."
if [ -d "./include/graph/generated" ]; then
    generated_count=$(find ./include/graph/generated -name "*.h" -o -name "*.java" | wc -l)
    echo "✅ Found $generated_count generated files in ./include/graph/generated"

    # List some example files
    find ./include/graph/generated -name "*.h" -o -name "*.java" | head -5 | while read -r file; do
        echo "   - $file"
    done
else
    echo "⚠️  Warning: Generated directory not found"
fi