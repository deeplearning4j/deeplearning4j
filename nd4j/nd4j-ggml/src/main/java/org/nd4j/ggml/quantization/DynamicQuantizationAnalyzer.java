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

package org.nd4j.ggml.quantization;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Analyzes tensors in a SameDiff model and assigns optimal quantization types
 * per tensor based on sensitivity analysis and a target size budget.
 *
 * <p>The analyzer works as follows:
 * <ol>
 *   <li>Iterates over all VARIABLE and CONSTANT tensors in the model</li>
 *   <li>Assigns fixed precision to embedding and output tensors</li>
 *   <li>Measures sensitivity of remaining tensors using the configured metric</li>
 *   <li>Ranks tensors by sensitivity (higher = more sensitive to quantization)</li>
 *   <li>Given the target size budget, allocates higher precision to more sensitive layers</li>
 * </ol>
 *
 * <p>Example usage:
 * <pre>{@code
 * DynamicQuantConfig config = DynamicQuantConfig.unslothDynamic2();
 * DynamicQuantizationAnalyzer analyzer = new DynamicQuantizationAnalyzer();
 * Map<String, GGMLDataType> assignments = analyzer.analyze(model, config);
 * }</pre>
 */
@Slf4j
public class DynamicQuantizationAnalyzer {

    /**
     * Internal record for tracking tensor information during analysis.
     */
    private static class TensorInfo {
        final String name;
        final INDArray array;
        final long numElements;
        double sensitivity;

        TensorInfo(String name, INDArray array) {
            this.name = name;
            this.array = array;
            this.numElements = array.length();
        }
    }

    /**
     * Analyze a SameDiff model and assign optimal quantization types per tensor.
     *
     * @param model  the SameDiff model to analyze
     * @param config the dynamic quantization configuration
     * @return a map from tensor name to assigned GGMLDataType
     */
    public Map<String, GGMLDataType> analyze(SameDiff model, DynamicQuantConfig config) {
        Map<String, GGMLDataType> assignments = new LinkedHashMap<>();
        List<TensorInfo> generalTensors = new ArrayList<>();

        // Categorize tensors
        for (SDVariable var : model.variables()) {
            if (var.getVariableType() != VariableType.VARIABLE &&
                var.getVariableType() != VariableType.CONSTANT) {
                continue;
            }

            INDArray arr = var.getArr();
            if (arr == null) {
                continue;
            }

            String name = var.name();

            // Check if this is an embedding or output tensor
            if (isEmbeddingTensor(name)) {
                assignments.put(name, config.getEmbeddingPrecision().toGGMLDataType());
                log.debug("Assigned embedding precision {} to tensor: {}",
                        config.getEmbeddingPrecision(), name);
            } else if (isOutputTensor(name)) {
                assignments.put(name, config.getEmbeddingPrecision().toGGMLDataType());
                log.debug("Assigned output precision {} to tensor: {}",
                        config.getEmbeddingPrecision(), name);
            } else if (isAttentionTensor(name)) {
                assignments.put(name, config.getAttentionPrecision().toGGMLDataType());
                log.debug("Assigned attention precision {} to tensor: {}",
                        config.getAttentionPrecision(), name);
            } else {
                generalTensors.add(new TensorInfo(name, arr));
            }
        }

        // Compute sensitivity for general tensors
        for (TensorInfo info : generalTensors) {
            info.sensitivity = computeSensitivity(info.array, config.getSensitivityMetric());
        }

        // Sort by sensitivity descending (most sensitive first)
        generalTensors.sort(Comparator.comparingDouble((TensorInfo t) -> t.sensitivity).reversed());

        // Assign quantization types based on budget
        List<ExportOptions.QuantizationType> candidates = config.getCandidateTypes();
        if (candidates == null || candidates.isEmpty()) {
            // Fallback: use min and max types only
            candidates = new ArrayList<>();
            candidates.add(config.getMinQuantType());
            candidates.add(config.getMaxQuantType());
        }

        // Sort candidates by quality (highest quality = highest bytes per element)
        candidates.sort(Comparator.comparingDouble(
                (ExportOptions.QuantizationType t) -> t.toGGMLDataType().getBytesPerElement()).reversed());

        if (config.getTargetModelSizeMB() > 0) {
            assignWithBudget(generalTensors, candidates, config, assignments);
        } else {
            assignByRank(generalTensors, candidates, assignments);
        }

        log.info("Dynamic quantization analysis complete: {} total tensors assigned", assignments.size());
        return assignments;
    }

    /**
     * Assign quantization types with a target size budget.
     * Starts with the lowest quality (smallest) type for all tensors,
     * then upgrades the most sensitive tensors until the budget is exhausted.
     */
    private void assignWithBudget(List<TensorInfo> tensors,
                                  List<ExportOptions.QuantizationType> candidates,
                                  DynamicQuantConfig config,
                                  Map<String, GGMLDataType> assignments) {
        long targetBytes = config.getTargetModelSizeMB() * 1024 * 1024;

        // Start with minimum quality for all
        ExportOptions.QuantizationType minType = candidates.get(candidates.size() - 1);
        Map<String, ExportOptions.QuantizationType> currentAssignments = new LinkedHashMap<>();
        for (TensorInfo info : tensors) {
            currentAssignments.put(info.name, minType);
        }

        // Calculate current total size
        long currentSize = calculateTotalSize(tensors, currentAssignments);

        // Upgrade most sensitive tensors one at a time
        for (TensorInfo info : tensors) {
            if (currentSize >= targetBytes) {
                break;
            }

            // Find the best upgrade that fits within budget
            ExportOptions.QuantizationType currentType = currentAssignments.get(info.name);
            for (ExportOptions.QuantizationType candidate : candidates) {
                if (candidate.toGGMLDataType().getBytesPerElement() <= currentType.toGGMLDataType().getBytesPerElement()) {
                    continue;
                }

                long sizeDelta = calculateSizeDelta(info, currentType, candidate);
                if (currentSize + sizeDelta <= targetBytes) {
                    currentAssignments.put(info.name, candidate);
                    currentSize += sizeDelta;
                    log.debug("Upgraded tensor {} from {} to {} (sensitivity={}, size now {}MB)",
                            info.name, currentType, candidate, info.sensitivity,
                            currentSize / (1024 * 1024));
                    break;
                }
            }
        }

        // Convert to GGMLDataType assignments
        for (Map.Entry<String, ExportOptions.QuantizationType> entry : currentAssignments.entrySet()) {
            assignments.put(entry.getKey(), entry.getValue().toGGMLDataType());
        }
    }

    /**
     * Assign quantization types by rank without a size budget.
     * Distributes candidate types evenly across tensors sorted by sensitivity.
     * Most sensitive tensors get the highest quality type.
     */
    private void assignByRank(List<TensorInfo> tensors,
                              List<ExportOptions.QuantizationType> candidates,
                              Map<String, GGMLDataType> assignments) {
        if (tensors.isEmpty()) {
            return;
        }

        int numCandidates = candidates.size();
        int tensorsPerBucket = Math.max(1, tensors.size() / numCandidates);

        for (int i = 0; i < tensors.size(); i++) {
            int bucketIndex = Math.min(i / tensorsPerBucket, numCandidates - 1);
            ExportOptions.QuantizationType type = candidates.get(bucketIndex);
            assignments.put(tensors.get(i).name, type.toGGMLDataType());

            log.debug("Rank-assigned tensor {} to {} (sensitivity={}, bucket={})",
                    tensors.get(i).name, type, tensors.get(i).sensitivity, bucketIndex);
        }
    }

    /**
     * Compute sensitivity of a tensor using the specified metric.
     *
     * @param array  the tensor data
     * @param metric the sensitivity metric to use
     * @return sensitivity score (higher = more sensitive to quantization)
     */
    private double computeSensitivity(INDArray array, DynamicQuantConfig.SensitivityMetric metric) {
        switch (metric) {
            case SNR:
                return computeSNR(array);
            case MAE:
                return computeMAE(array);
            case WEIGHT_MAGNITUDE:
            default:
                return computeWeightMagnitude(array);
        }
    }

    /**
     * Compute signal-to-noise ratio.
     * Higher SNR means the tensor has more signal relative to noise,
     * making it more sensitive to quantization errors.
     */
    private double computeSNR(INDArray array) {
        INDArray floatArr = array.castTo(DataType.FLOAT);
        double mean = floatArr.meanNumber().doubleValue();
        double variance = floatArr.varNumber().doubleValue();
        if (variance < 1e-10) {
            return 0.0;
        }
        return (mean * mean) / variance;
    }

    /**
     * Compute mean absolute value of the tensor.
     * Higher values indicate larger weights that are more sensitive to quantization.
     */
    private double computeMAE(INDArray array) {
        INDArray floatArr = array.castTo(DataType.FLOAT);
        return Nd4j.math().abs(floatArr).meanNumber().doubleValue();
    }

    /**
     * Compute weight magnitude (L2 norm normalized by element count).
     * Larger magnitudes indicate more important weights.
     */
    private double computeWeightMagnitude(INDArray array) {
        INDArray floatArr = array.castTo(DataType.FLOAT);
        double norm2 = floatArr.norm2Number().doubleValue();
        long numElements = array.length();
        if (numElements == 0) {
            return 0.0;
        }
        return norm2 / Math.sqrt(numElements);
    }

    /**
     * Check if a tensor name indicates an embedding layer.
     */
    private boolean isEmbeddingTensor(String name) {
        String lower = name.toLowerCase();
        return lower.contains("embed") ||
               lower.contains("token_embd") ||
               lower.contains("wte") ||
               lower.contains("word_embedding");
    }

    /**
     * Check if a tensor name indicates an output/language model head tensor.
     */
    private boolean isOutputTensor(String name) {
        String lower = name.toLowerCase();
        return lower.contains("output") && lower.contains("weight") && !lower.contains("layer") ||
               lower.contains("lm_head") ||
               lower.contains("output.weight");
    }

    /**
     * Check if a tensor name indicates an attention layer tensor.
     */
    private boolean isAttentionTensor(String name) {
        String lower = name.toLowerCase();
        return lower.contains("attn") ||
               lower.contains("attention") ||
               lower.contains("self_attn") ||
               lower.contains("wq") ||
               lower.contains("wk") ||
               lower.contains("wv");
    }

    /**
     * Calculate total size in bytes for a set of tensor assignments.
     */
    private long calculateTotalSize(List<TensorInfo> tensors,
                                    Map<String, ExportOptions.QuantizationType> assignments) {
        long total = 0;
        for (TensorInfo info : tensors) {
            ExportOptions.QuantizationType type = assignments.get(info.name);
            total += type.toGGMLDataType().calculateStorageBytes(info.numElements);
        }
        return total;
    }

    /**
     * Calculate the size difference when upgrading a tensor from one type to another.
     */
    private long calculateSizeDelta(TensorInfo info,
                                    ExportOptions.QuantizationType fromType,
                                    ExportOptions.QuantizationType toType) {
        long fromSize = fromType.toGGMLDataType().calculateStorageBytes(info.numElements);
        long toSize = toType.toGGMLDataType().calculateStorageBytes(info.numElements);
        return toSize - fromSize;
    }
}
