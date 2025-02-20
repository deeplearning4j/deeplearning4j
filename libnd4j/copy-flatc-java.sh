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

SOURCE_DIR="include/graph/generated/org/nd4j/graph"
TARGET_DIR="../nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/graph"

# Enable debug output
set -x
echo "Source directory: $SOURCE_DIR"
echo "Target directory: $TARGET_DIR"
ls -la "$SOURCE_DIR" || echo "Source directory does not exist or is empty"

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

# Process each Java file
for file in "$SOURCE_DIR"/*.java; do
    if [ -f "$file" ]; then
        # Create temp file
        temp_file=$(mktemp)
        # Add copyright header
        echo "$COPYRIGHT" > "$temp_file"
        # Add a newline
        echo "" >> "$temp_file"
        # Append the original file content
        cat "$file" >> "$temp_file"
        # Move to final destination
        cp "$temp_file" "$TARGET_DIR/$(basename "$file")"
        # Clean up temp file
        rm "$temp_file"
    fi
done

echo "Java files copied to target location with copyright headers successfully"