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

package org.nd4j.torchscript.architecture;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.torchscript.convert.ConversionOptions;
import org.nd4j.torchscript.convert.TensorNameMapper;
import org.nd4j.torchscript.format.TorchScriptMetadata;
import org.nd4j.torchscript.format.TorchScriptTensorInfo;

import java.util.*;

/**
 * Fallback architecture handler for unrecognized CNN models.
 * Attempts to build a graph by analyzing tensor names and shapes.
 */
@Slf4j
public class GenericCNNArchitecture implements VisionArchitecture {

    @Override
    public String getName() {
        return "generic_cnn";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return Collections.singleton("generic_cnn");
    }

    @Override
    public boolean canHandle(TorchScriptMetadata metadata) {
        // Always returns true as fallback
        return true;
    }

    @Override
    public SameDiff buildGraph(TorchScriptMetadata metadata, Map<String, INDArray> weights,
                               ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);

        log.info("Building generic CNN graph: {} tensors", metadata.getTensors().size());

        // Input placeholder
        SDVariable input = sd.placeHolder("input", DataType.FLOAT,
                -1, config.getInputChannels(), -1, -1);

        // Group weights by layer
        Map<String, List<TorchScriptTensorInfo>> layerWeights = groupByLayer(metadata.getTensors());

        log.info("Detected {} layers", layerWeights.size());

        // Build layers in order
        SDVariable x = input;
        List<String> sortedLayers = new ArrayList<>(layerWeights.keySet());
        Collections.sort(sortedLayers);

        for (String layerName : sortedLayers) {
            List<TorchScriptTensorInfo> params = layerWeights.get(layerName);
            x = buildLayer(sd, x, layerName, params, weights);
        }

        // Mark output
        if (x != null) {
            sd.setOutputs(x.name());
        }

        return sd;
    }

    private Map<String, List<TorchScriptTensorInfo>> groupByLayer(List<TorchScriptTensorInfo> tensors) {
        Map<String, List<TorchScriptTensorInfo>> groups = new LinkedHashMap<>();

        for (TorchScriptTensorInfo tensor : tensors) {
            String layerName = TensorNameMapper.extractLayerName(tensor.getName());
            groups.computeIfAbsent(layerName, k -> new ArrayList<>()).add(tensor);
        }

        return groups;
    }

    private SDVariable buildLayer(SameDiff sd, SDVariable input, String layerName,
                                   List<TorchScriptTensorInfo> params, Map<String, INDArray> weights) {
        SDVariable x = input;

        // Infer layer type from parameter names/shapes
        String layerType = inferLayerType(params);

        switch (layerType) {
            case "conv2d":
                x = buildConv2d(sd, x, layerName, params, weights);
                break;
            case "batch_norm":
                x = buildBatchNorm(sd, x, layerName, params, weights);
                break;
            case "linear":
                x = buildLinear(sd, x, layerName, params, weights);
                break;
            default:
                log.debug("Unknown layer type for {}: storing weights only", layerName);
                storeWeights(sd, layerName, params, weights);
                break;
        }

        return x;
    }

    private String inferLayerType(List<TorchScriptTensorInfo> params) {
        for (TorchScriptTensorInfo param : params) {
            String name = param.getName().toLowerCase();
            long[] shape = param.getShape();

            if (name.endsWith(".weight") || name.endsWith("_weight")) {
                if (shape != null) {
                    if (shape.length == 4) {
                        return "conv2d";
                    } else if (shape.length == 2) {
                        return "linear";
                    } else if (shape.length == 1 && name.contains("bn")) {
                        return "batch_norm";
                    }
                }
            }
            if (name.contains("running_mean") || name.contains("running_var")) {
                return "batch_norm";
            }
        }

        return "unknown";
    }

    private SDVariable buildConv2d(SameDiff sd, SDVariable input, String layerName,
                                    List<TorchScriptTensorInfo> params, Map<String, INDArray> weights) {
        log.debug("Building conv2d layer: {}", layerName);

        // Get weight and bias
        INDArray weightArr = findWeight(layerName, "weight", weights);
        INDArray biasArr = findWeight(layerName, "bias", weights);

        if (weightArr != null) {
            String wName = TensorNameMapper.normalizeName(layerName) + "_W";
            sd.constant(wName, weightArr);
        }

        if (biasArr != null) {
            String bName = TensorNameMapper.normalizeName(layerName) + "_b";
            sd.constant(bName, biasArr);
        }

        // Note: Actual conv2d would be created here with proper config
        return input;
    }

    private SDVariable buildBatchNorm(SameDiff sd, SDVariable input, String layerName,
                                       List<TorchScriptTensorInfo> params, Map<String, INDArray> weights) {
        log.debug("Building batch_norm layer: {}", layerName);

        INDArray gamma = findWeight(layerName, "weight", weights);
        INDArray beta = findWeight(layerName, "bias", weights);
        INDArray runningMean = findWeight(layerName, "running_mean", weights);
        INDArray runningVar = findWeight(layerName, "running_var", weights);

        String prefix = TensorNameMapper.normalizeName(layerName);
        if (gamma != null) sd.constant(prefix + "_gamma", gamma);
        if (beta != null) sd.constant(prefix + "_beta", beta);
        if (runningMean != null) sd.constant(prefix + "_mean", runningMean);
        if (runningVar != null) sd.constant(prefix + "_var", runningVar);

        return input;
    }

    private SDVariable buildLinear(SameDiff sd, SDVariable input, String layerName,
                                    List<TorchScriptTensorInfo> params, Map<String, INDArray> weights) {
        log.debug("Building linear layer: {}", layerName);

        INDArray weightArr = findWeight(layerName, "weight", weights);
        INDArray biasArr = findWeight(layerName, "bias", weights);

        if (weightArr != null) {
            String wName = TensorNameMapper.normalizeName(layerName) + "_W";
            SDVariable w = sd.constant(wName, weightArr);

            if (biasArr != null) {
                String bName = TensorNameMapper.normalizeName(layerName) + "_b";
                SDVariable b = sd.constant(bName, biasArr);
                return sd.nn().linear(input, w, b);
            }
        }

        return input;
    }

    private void storeWeights(SameDiff sd, String layerName, List<TorchScriptTensorInfo> params,
                              Map<String, INDArray> weights) {
        for (TorchScriptTensorInfo param : params) {
            String mappedName = TensorNameMapper.normalizeName(param.getName());
            if (weights.containsKey(mappedName)) {
                sd.constant(mappedName, weights.get(mappedName));
            } else if (weights.containsKey(param.getName())) {
                sd.constant(mappedName, weights.get(param.getName()));
            }
        }
    }

    private INDArray findWeight(String layerName, String paramType, Map<String, INDArray> weights) {
        // Try different naming conventions
        String[] candidates = {
                layerName + "." + paramType,
                layerName + "_" + paramType,
                TensorNameMapper.normalizeName(layerName) + "_" + paramType,
                TensorNameMapper.normalizeName(layerName + "." + paramType)
        };

        for (String candidate : candidates) {
            if (weights.containsKey(candidate)) {
                return weights.get(candidate);
            }
        }

        return null;
    }

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();
        patterns.put("{layer}.weight", "Layer weights");
        patterns.put("{layer}.bias", "Layer bias");
        patterns.put("{layer}.running_mean", "Batch norm running mean");
        patterns.put("{layer}.running_var", "Batch norm running variance");
        return patterns;
    }

    @Override
    public int getPriority() {
        return -1; // Lowest priority - fallback only
    }
}
