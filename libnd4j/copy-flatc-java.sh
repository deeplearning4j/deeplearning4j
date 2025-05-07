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
#  * distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#

# Exit on error
set -e

# Determine script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find the project structure
# First try to detect libnd4j directory
if [[ "$SCRIPT_DIR" == */libnd4j* ]]; then
    # We're already in or under libnd4j directory
    LIBND4J_DIR=$(echo "$SCRIPT_DIR" | sed 's/\(.*libnd4j\).*/\1/')
    PROJECT_ROOT=$(dirname "$LIBND4J_DIR")
else
    # Check if we're in the parent deeplearning4j directory
    if [ -d "$SCRIPT_DIR/libnd4j" ]; then
        LIBND4J_DIR="$SCRIPT_DIR/libnd4j"
        PROJECT_ROOT="$SCRIPT_DIR"
    else
        # Try going up one directory to see if libnd4j is there
        if [ -d "$SCRIPT_DIR/../libnd4j" ]; then
            LIBND4J_DIR="$SCRIPT_DIR/../libnd4j"
            PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
        else
            echo "Error: Could not locate libnd4j directory"
            exit 1
        fi
    fi
fi

# Set the absolute paths for source and target
SOURCE_DIR="$LIBND4J_DIR/include/graph/generated/org/nd4j/graph"
TARGET_DIR="$PROJECT_ROOT/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/graph"

# Enable debug output
set -x
echo "Script directory: $SCRIPT_DIR"
echo "LibND4J directory: $LIBND4J_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Source directory: $SOURCE_DIR"
echo "Target directory: $TARGET_DIR"

# Check if source directory exists and has content
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# List source directory content
ls -la "$SOURCE_DIR" || {
    echo "Error: Cannot access source directory or it is empty"
    exit 1
}

# Create copyright header content
COPYRIGHT="/*
 * ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
"

# Ensure target directory exists
mkdir -p "$TARGET_DIR"

# Check if any Java files exist in source directory
if ! ls "$SOURCE_DIR"/*.java &> /dev/null; then
    echo "Error: No Java files found in source directory: $SOURCE_DIR"
    exit 1
fi

# Process each Java file
for file in "$SOURCE_DIR"/*.java; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        target_file="$TARGET_DIR/$filename"
        echo "Processing: $filename"

        # Create temp file
        temp_file=$(mktemp)
        # Add copyright header
        echo "$COPYRIGHT" > "$temp_file"
        # Add a newline
        echo "" >> "$temp_file"
        # Append the original file content
        cat "$file" >> "$temp_file"
        # Move to final destination
        cp "$temp_file" "$target_file"
        # Clean up temp file
        rm "$temp_file"

        echo "Created: $target_file"
    fi
done

echo "Java files copied to target location with copyright headers successfully"