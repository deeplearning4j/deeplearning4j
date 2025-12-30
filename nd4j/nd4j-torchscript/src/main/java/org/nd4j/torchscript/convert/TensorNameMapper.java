/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.torchscript.convert;

import java.util.Map;
import java.util.regex.Pattern;

/**
 * Utility for mapping PyTorch tensor names to SameDiff variable names.
 */
public class TensorNameMapper {

    // Pattern to match invalid characters in SameDiff variable names
    private static final Pattern INVALID_CHARS = Pattern.compile("[^a-zA-Z0-9_]");

    private TensorNameMapper() {
        // Utility class
    }

    /**
     * Map a PyTorch tensor name to a SameDiff-compatible name.
     *
     * @param torchName the PyTorch tensor name
     * @param customMappings custom name mappings (may be null)
     * @return the mapped name
     */
    public static String mapName(String torchName, Map<String, String> customMappings) {
        // Check custom mappings first
        if (customMappings != null && customMappings.containsKey(torchName)) {
            return customMappings.get(torchName);
        }

        // Normalize the name
        return normalizeName(torchName);
    }

    /**
     * Normalize a tensor name to be SameDiff-compatible.
     * Replaces dots with underscores and removes invalid characters.
     *
     * @param name the original name
     * @return normalized name
     */
    public static String normalizeName(String name) {
        if (name == null) {
            return "unnamed";
        }

        // Replace dots with underscores (PyTorch uses dots for hierarchy)
        String normalized = name.replace(".", "_");

        // Replace any remaining invalid characters
        normalized = INVALID_CHARS.matcher(normalized).replaceAll("_");

        // Ensure doesn't start with a number
        if (!normalized.isEmpty() && Character.isDigit(normalized.charAt(0))) {
            normalized = "t_" + normalized;
        }

        return normalized;
    }

    /**
     * Check if a tensor name represents a weight tensor.
     *
     * @param name the tensor name
     * @return true if this is a weight tensor
     */
    public static boolean isWeightTensor(String name) {
        return name.endsWith(".weight") || name.endsWith("_weight");
    }

    /**
     * Check if a tensor name represents a bias tensor.
     *
     * @param name the tensor name
     * @return true if this is a bias tensor
     */
    public static boolean isBiasTensor(String name) {
        return name.endsWith(".bias") || name.endsWith("_bias");
    }

    /**
     * Check if a tensor name represents a batch normalization buffer.
     *
     * @param name the tensor name
     * @return true if this is a batch norm buffer
     */
    public static boolean isBatchNormBuffer(String name) {
        String lower = name.toLowerCase();
        return lower.contains("running_mean") ||
               lower.contains("running_var") ||
               lower.contains("num_batches_tracked");
    }

    /**
     * Extract the layer name from a parameter name.
     * For example, "layer1.0.conv1.weight" -> "layer1.0.conv1"
     *
     * @param paramName the parameter name
     * @return the layer name
     */
    public static String extractLayerName(String paramName) {
        int lastDot = paramName.lastIndexOf('.');
        if (lastDot > 0) {
            return paramName.substring(0, lastDot);
        }
        return paramName;
    }

    /**
     * Extract the parameter type from a name.
     * For example, "layer1.0.conv1.weight" -> "weight"
     *
     * @param paramName the parameter name
     * @return the parameter type
     */
    public static String extractParamType(String paramName) {
        int lastDot = paramName.lastIndexOf('.');
        if (lastDot >= 0 && lastDot < paramName.length() - 1) {
            return paramName.substring(lastDot + 1);
        }
        return paramName;
    }

    /**
     * Get the SameDiff weight key for a layer.
     *
     * @param layerName the layer name
     * @return the weight variable name
     */
    public static String getWeightKey(String layerName) {
        return normalizeName(layerName) + "_W";
    }

    /**
     * Get the SameDiff bias key for a layer.
     *
     * @param layerName the layer name
     * @return the bias variable name
     */
    public static String getBiasKey(String layerName) {
        return normalizeName(layerName) + "_b";
    }

    /**
     * Infer the layer type from a parameter name.
     *
     * @param paramName the parameter name
     * @return inferred layer type or "unknown"
     */
    public static String inferLayerType(String paramName) {
        String lower = paramName.toLowerCase();

        if (lower.contains("conv")) {
            if (lower.contains("conv1d")) return "conv1d";
            if (lower.contains("conv3d")) return "conv3d";
            return "conv2d";
        }
        if (lower.contains("bn") || lower.contains("batch_norm") || lower.contains("batchnorm")) {
            return "batch_norm";
        }
        if (lower.contains("fc") || lower.contains("linear") || lower.contains("dense")) {
            return "linear";
        }
        if (lower.contains("lstm")) return "lstm";
        if (lower.contains("gru")) return "gru";
        if (lower.contains("embedding")) return "embedding";
        if (lower.contains("layernorm") || lower.contains("layer_norm")) return "layer_norm";
        if (lower.contains("groupnorm") || lower.contains("group_norm")) return "group_norm";

        return "unknown";
    }
}
