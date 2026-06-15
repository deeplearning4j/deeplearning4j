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
 * Per-layer adaptive quantizer for GGUF export.
 *
 * <p>Analyzes layer sensitivity via KL divergence between full-precision (FP32) and
 * quantized layer outputs on a calibration dataset. Assigns per-layer quantization
 * types so that sensitive layers receive higher precision while insensitive layers
 * are compressed more aggressively.
 *
 * <p>Algorithm overview:
 * <ol>
 *   <li>For each weight tensor W, compute the reference output {@code Y = X @ W.T}
 *       using FP32 weights and calibration data X.</li>
 *   <li>Quantize W to the candidate type, dequantize back to FP32, and compute
 *       {@code Y_q = X @ dequant(quant(W)).T}.</li>
 *   <li>Compute KL divergence KL(softmax(Y) || softmax(Y_q)) as the sensitivity
 *       score for this layer.</li>
 *   <li>Rank layers by sensitivity. Layers in the top
 *       {@link AdaptiveQuantConfig#getHighSensitivityPercentile()} fraction receive
 *       the highest quality type; layers in the bottom
 *       {@link AdaptiveQuantConfig#getLowSensitivityPercentile()} fraction receive
 *       the most aggressive type. Middle layers are distributed across the remaining
 *       candidate types.</li>
 *   <li>Layers named in {@link AdaptiveQuantConfig#getPreserveLayers()} always
 *       receive the highest quality type regardless of their sensitivity score.</li>
 * </ol>
 *
 * <p>If {@link AdaptiveQuantConfig#getTargetBitsPerWeight()} is non-zero the
 * assignment iterates sensitivity-rank order and upgrades layers until the average
 * bits-per-weight target is met.
 *
 * <p>Example usage:
 * <pre>{@code
 * AdaptiveQuantConfig config = AdaptiveQuantConfig.balanced();
 * AdaptiveLayerQuantizer quantizer = new AdaptiveLayerQuantizer();
 * INDArray calibData = Nd4j.randn(128, 2048, 4096);
 * Map<String, GGMLQuantType> plan = quantizer.analyzeAndAssign(weights, calibData, config);
 * }</pre>
 *
 * @see AdaptiveQuantConfig
 * @see GGMLQuantType
 */
@Slf4j
public class AdaptiveLayerQuantizer {

    /**
     * Analyze all weight layers and produce a per-layer quantization assignment plan.
     *
     * @param weights       map of layer name to weight tensor (2-D expected: [out, in])
     * @param calibrationData calibration inputs with shape [numSamples, seqLen, hidden] or
     *                        [numSamples, hidden]; the last dimension must match the weight
     *                        inner dimension
     * @param config        adaptive quantization configuration
     * @return map of layer name to assigned {@link GGMLQuantType}; never null
     * @throws IllegalArgumentException if config validation fails
     */
    public Map<String, GGMLQuantType> analyzeAndAssign(
            Map<String, INDArray> weights,
            INDArray calibrationData,
            AdaptiveQuantConfig config) {

        config.validate();

        Map<String, GGMLQuantType> result = new LinkedHashMap<>();

        // Assign preserve layers first
        for (String name : weights.keySet()) {
            if (config.getPreserveLayers().contains(name)) {
                result.put(name, config.highestQualityType());
                log.debug("Preserved layer {} at {}", name, config.highestQualityType());
            }
        }

        // Compute sensitivities for non-preserved layers
        List<String> toAnalyze = new ArrayList<>();
        for (String name : weights.keySet()) {
            if (!result.containsKey(name)) {
                toAnalyze.add(name);
            }
        }

        Map<String, Double> sensitivities = computeLayerSensitivities(
                filterMap(weights, toAnalyze), calibrationData);

        // Sort layers by sensitivity descending (most sensitive first)
        toAnalyze.sort(Comparator.comparingDouble(
                (String n) -> sensitivities.getOrDefault(n, 0.0)).reversed());

        int n = toAnalyze.size();
        if (n == 0) {
            return result;
        }

        List<GGMLQuantType> types = config.getQuantTypes();
        int numTypes = types.size();

        if (config.getTargetBitsPerWeight() > 0) {
            assignWithBudget(toAnalyze, weights, sensitivities, types, config, result);
        } else {
            assignByRank(toAnalyze, sensitivities, types, config, result);
        }

        log.info("Adaptive quantization plan: {} layers assigned ({} preserved)",
                result.size(), result.size() - toAnalyze.size());
        return result;
    }

    /**
     * Compute per-layer KL divergence sensitivities.
     *
     * <p>For each layer, the method:
     * <ol>
     *   <li>Flattens the calibration data to [rows, inDim] (matching the weight's inner dim).</li>
     *   <li>Computes reference logits: {@code Y = X @ W.T}.</li>
     *   <li>Quantizes W using the lowest-quality type in the configured candidate list,
     *       then dequantizes back to FP32.</li>
     *   <li>Computes quantized logits: {@code Y_q = X @ W_q.T}.</li>
     *   <li>Returns mean KL(softmax(Y) || softmax(Y_q)) over all rows.</li>
     * </ol>
     *
     * @param weights       map of layer name to weight tensor
     * @param calibrationData calibration inputs; last dimension is the hidden dimension
     * @return map of layer name to KL divergence sensitivity score
     */
    public Map<String, Double> computeLayerSensitivities(
            Map<String, INDArray> weights,
            INDArray calibrationData) {

        Map<String, Double> sensitivities = new LinkedHashMap<>();

        // Flatten calibration data to 2D: [rows, inDim]
        INDArray calib2d = flattenTo2D(calibrationData.castTo(DataType.FLOAT));

        for (Map.Entry<String, INDArray> entry : weights.entrySet()) {
            String name = entry.getKey();
            INDArray weight = entry.getValue().castTo(DataType.FLOAT);

            if (weight.rank() < 2) {
                // Skip 1-D biases / scalars — they are not matrix-multiplied
                sensitivities.put(name, 0.0);
                continue;
            }

            long inDim = weight.size(weight.rank() - 1);

            // Trim / reshape calib rows to match inDim
            INDArray calibInput = prepareCalibInput(calib2d, inDim);
            if (calibInput == null) {
                // Incompatible shapes — assign neutral sensitivity
                sensitivities.put(name, 0.0);
                continue;
            }

            try {
                double kl = computeKLSensitivity(calibInput, weight);
                sensitivities.put(name, kl);
                log.debug("Layer {} sensitivity (KL): {}", name, kl);
            } catch (Exception e) {
                log.warn("Could not compute sensitivity for layer {}: {}", name, e.getMessage());
                sensitivities.put(name, 0.0);
            }
        }

        return sensitivities;
    }

    // ========== Private helpers ==========

    /**
     * Compute KL(softmax(Y) || softmax(Y_q)) for a single weight matrix.
     *
     * @param calibInput 2D calibration input [rows, inDim]
     * @param weight     weight matrix [outDim, inDim]
     * @return mean KL divergence over all rows
     */
    private double computeKLSensitivity(INDArray calibInput, INDArray weight) {
        // Reference output: Y = calibInput @ weight.T  -> [rows, outDim]
        INDArray refOutput = calibInput.mmul(weight.transpose());

        // Quantize weight and dequantize back
        INDArray quantizedWeight = quantizeDequantize(weight);

        // Quantized output: Y_q = calibInput @ quantizedWeight.T -> [rows, outDim]
        INDArray quantOutput = calibInput.mmul(quantizedWeight.transpose());

        // KL divergence per row, then average
        return meanKLDivergence(refOutput, quantOutput);
    }

    /**
     * Round-trip quantize then dequantize a weight matrix using Q4_K (as the probe
     * quantization type for sensitivity measurement).
     */
    private INDArray quantizeDequantize(INDArray weight) {
        float[] floatData = flattenFloat(weight);

        // Use Q4_K as the measurement quant type — a reasonable midpoint proxy
        GGMLDataType probeType = org.nd4j.ggml.format.GGMLDataType.GGML_TYPE_Q4_K;

        if (!QuantizerFactory.hasQuantizer(probeType) ||
                !DequantizerFactory.hasDequantizer(probeType)) {
            // Fallback: no round-trip, return weight as-is (zero sensitivity)
            return weight;
        }

        byte[] quantized = QuantizerFactory.quantize(floatData, probeType);
        float[] dequantized = DequantizerFactory.getDequantizer(probeType)
                .dequantize(quantized, floatData.length);

        INDArray result = Nd4j.create(dequantized).reshape(weight.shape()).castTo(DataType.FLOAT);
        return result;
    }

    /**
     * Compute mean KL(softmax(P) || softmax(Q)) over rows.
     * KL(P||Q) = sum_i P_i * log(P_i / Q_i).
     * Uses log-sum-exp stabilisation.
     */
    private double meanKLDivergence(INDArray p, INDArray q) {
        long rows = p.size(0);
        long cols = p.size(1);
        double totalKL = 0.0;

        for (long r = 0; r < rows; r++) {
            // Numerically stable softmax for row r of P and Q
            double[] pRow = rowToDouble(p, r, cols);
            double[] qRow = rowToDouble(q, r, cols);

            double[] pSoft = softmax(pRow);
            double[] qSoft = softmax(qRow);

            double kl = 0.0;
            for (int c = 0; c < cols; c++) {
                double pi = pSoft[c];
                double qi = qSoft[c];
                if (pi > 1e-12 && qi > 1e-12) {
                    kl += pi * Math.log(pi / qi);
                }
            }
            totalKL += kl;
        }

        return rows > 0 ? totalKL / rows : 0.0;
    }

    /**
     * Numerically stable softmax.
     */
    private double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logits) {
            if (v > max) max = v;
        }
        double sum = 0.0;
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = Math.exp(logits[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < logits.length; i++) {
            out[i] /= sum;
        }
        return out;
    }

    private double[] rowToDouble(INDArray arr, long row, long cols) {
        double[] out = new double[(int) cols];
        for (int c = 0; c < cols; c++) {
            out[c] = arr.getDouble(row, c);
        }
        return out;
    }

    /**
     * Assign quant types using a bits-per-weight budget.
     * Starts all non-preserved layers at the lowest quality, then upgrades the most
     * sensitive layers one step at a time until the average bpw target is reached.
     */
    private void assignWithBudget(
            List<String> sortedLayers,
            Map<String, INDArray> weights,
            Map<String, Double> sensitivities,
            List<GGMLQuantType> types,
            AdaptiveQuantConfig config,
            Map<String, GGMLQuantType> result) {

        int n = sortedLayers.size();
        GGMLQuantType lowestType = config.lowestQualityType();
        GGMLQuantType highestType = config.highestQualityType();

        // Working assignment — start at lowest quality
        Map<String, GGMLQuantType> working = new LinkedHashMap<>();
        for (String name : sortedLayers) {
            working.put(name, lowestType);
        }

        // Always give top-percentile layers the highest quality immediately
        int highCount = (int) Math.ceil(n * config.getHighSensitivityPercentile());
        int lowCount = (int) Math.ceil(n * config.getLowSensitivityPercentile());
        for (int i = 0; i < highCount && i < n; i++) {
            working.put(sortedLayers.get(i), highestType);
        }

        // Compute current average bpw
        double currentBpw = averageBpw(sortedLayers, weights, working);
        double targetBpw = config.getTargetBitsPerWeight();

        // Upgrade middle layers until we hit the target (if current < target)
        if (currentBpw < targetBpw) {
            for (int i = highCount; i < n - lowCount; i++) {
                String name = sortedLayers.get(i);
                GGMLQuantType current = working.get(name);
                // Try each upgrade step
                for (GGMLQuantType candidate : types) {
                    if (candidate.getBitsPerWeight() <= current.getBitsPerWeight()) {
                        continue;
                    }
                    working.put(name, candidate);
                    double newBpw = averageBpw(sortedLayers, weights, working);
                    if (newBpw > targetBpw) {
                        // Overshot — revert and stop upgrading this layer
                        working.put(name, current);
                        break;
                    }
                    currentBpw = newBpw;
                    if (currentBpw >= targetBpw) {
                        break;
                    }
                }
                if (currentBpw >= targetBpw) {
                    break;
                }
            }
        }

        result.putAll(working);
    }

    /**
     * Assign quant types by sensitivity rank without a bits budget.
     * Top {@link AdaptiveQuantConfig#getHighSensitivityPercentile()} get highest quality;
     * bottom {@link AdaptiveQuantConfig#getLowSensitivityPercentile()} get lowest quality;
     * middle layers are distributed evenly across remaining candidate types.
     */
    private void assignByRank(
            List<String> sortedLayers,
            Map<String, Double> sensitivities,
            List<GGMLQuantType> types,
            AdaptiveQuantConfig config,
            Map<String, GGMLQuantType> result) {

        int n = sortedLayers.size();
        GGMLQuantType highestType = config.highestQualityType();
        GGMLQuantType lowestType = config.lowestQualityType();

        int highCount = (int) Math.ceil(n * config.getHighSensitivityPercentile());
        int lowCount = (int) Math.ceil(n * config.getLowSensitivityPercentile());
        int middleCount = n - highCount - lowCount;

        // Top layers
        for (int i = 0; i < highCount && i < n; i++) {
            result.put(sortedLayers.get(i), highestType);
        }

        // Middle layers — evenly distribute across middle candidates
        List<GGMLQuantType> middleTypes = types.size() > 2
                ? types.subList(1, types.size() - 1)
                : types;
        int numMiddleTypes = Math.max(1, middleTypes.size());
        for (int i = highCount; i < highCount + middleCount && i < n; i++) {
            int typeIdx = ((i - highCount) * numMiddleTypes) / Math.max(1, middleCount);
            typeIdx = Math.min(typeIdx, numMiddleTypes - 1);
            result.put(sortedLayers.get(i), middleTypes.get(typeIdx));
        }

        // Bottom layers
        for (int i = highCount + middleCount; i < n; i++) {
            result.put(sortedLayers.get(i), lowestType);
        }
    }

    /**
     * Compute weighted average bits-per-weight given per-layer assignments.
     * Weight is proportional to number of elements in each tensor.
     */
    private double averageBpw(
            List<String> layers,
            Map<String, INDArray> weights,
            Map<String, GGMLQuantType> assignments) {
        long totalElements = 0;
        double totalBits = 0.0;
        for (String name : layers) {
            INDArray w = weights.get(name);
            if (w == null) continue;
            long elems = w.length();
            totalElements += elems;
            totalBits += assignments.get(name).getBitsPerWeight() * elems;
        }
        return totalElements > 0 ? totalBits / totalElements : 0.0;
    }

    /**
     * Flatten an N-D array to 2D by collapsing all but the last dimension.
     */
    private static INDArray flattenTo2D(INDArray arr) {
        if (arr.rank() == 2) {
            return arr;
        }
        long lastDim = arr.size(arr.rank() - 1);
        long rows = arr.length() / lastDim;
        return arr.reshape(rows, lastDim);
    }

    /**
     * Prepare calibration input for a given inner dimension.
     * If the calibration data has more columns than inDim, truncate.
     * If it has fewer, return null (incompatible).
     */
    private static INDArray prepareCalibInput(INDArray calib2d, long inDim) {
        long calibCols = calib2d.size(1);
        if (calibCols == inDim) {
            return calib2d;
        }
        if (calibCols > inDim) {
            // Truncate to inDim
            return calib2d.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(0, inDim));
        }
        // calibCols < inDim — cannot use this calibration data for this layer
        return null;
    }

    /**
     * Flatten INDArray to float array.
     */
    private static float[] flattenFloat(INDArray arr) {
        INDArray flat = arr.reshape(arr.length()).castTo(DataType.FLOAT);
        return flat.toFloatVector();
    }

    /**
     * Return a filtered sub-map containing only the requested keys.
     */
    private static Map<String, INDArray> filterMap(Map<String, INDArray> map, List<String> keys) {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (String k : keys) {
            if (map.containsKey(k)) {
                result.put(k, map.get(k));
            }
        }
        return result;
    }
}
