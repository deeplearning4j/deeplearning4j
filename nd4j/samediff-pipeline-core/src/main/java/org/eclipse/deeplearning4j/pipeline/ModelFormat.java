/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.pipeline;

/**
 * Supported model formats for pipeline loading.
 */
public enum ModelFormat {
    SDZ("sdz", "SameDiff ZIP format"),
    SAFETENSORS("safetensors", "HuggingFace SafeTensors format"),
    GGUF("gguf", "GGML Universal Format"),
    ONNX("onnx", "Open Neural Network Exchange"),
    PYTORCH("bin", "PyTorch pickle format"),
    UNKNOWN("", "Unknown format");

    private final String extension;
    private final String description;

    ModelFormat(String extension, String description) {
        this.extension = extension;
        this.description = description;
    }

    public String getExtension() { return extension; }
    public String getDescription() { return description; }

    public static ModelFormat fromExtension(String ext) {
        if (ext == null) return UNKNOWN;
        String lower = ext.toLowerCase();
        for (ModelFormat format : values()) {
            if (format.extension.equalsIgnoreCase(lower)) {
                return format;
            }
        }
        // Additional mappings
        if (lower.equals("pt") || lower.equals("pth")) return PYTORCH;
        return UNKNOWN;
    }

    public static ModelFormat fromFilename(String filename) {
        if (filename == null) return UNKNOWN;
        String lower = filename.toLowerCase();
        if (lower.endsWith(".sdz")) return SDZ;
        if (lower.endsWith(".safetensors")) return SAFETENSORS;
        if (lower.endsWith(".gguf") || lower.endsWith(".ggml")) return GGUF;
        if (lower.endsWith(".onnx")) return ONNX;
        if (lower.endsWith(".bin") || lower.endsWith(".pt") || lower.endsWith(".pth")) return PYTORCH;
        return UNKNOWN;
    }
}
