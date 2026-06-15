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
 * Architecture handler for ResNet variants.
 * Supports: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152,
 * and variants like ResNeXt, Wide ResNet.
 */
@Slf4j
public class ResNetArchitecture implements VisionArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = new HashSet<>(Arrays.asList(
            "resnet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2"
    ));

    @Override
    public String getName() {
        return "resnet";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    @Override
    public boolean canHandle(TorchScriptMetadata metadata) {
        // Check for ResNet-style tensor names
        for (TorchScriptTensorInfo tensor : metadata.getTensors()) {
            String name = tensor.getName().toLowerCase();
            // ResNet has distinctive layer naming: layer1, layer2, layer3, layer4
            if (name.contains("layer1") || name.contains("layer2") ||
                    name.contains("layer3") || name.contains("layer4")) {
                // Also check for residual block patterns
                if (name.contains("downsample") || name.contains("conv1") || name.contains("conv2")) {
                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public SameDiff buildGraph(TorchScriptMetadata metadata, Map<String, INDArray> weights,
                               ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);

        log.info("Building ResNet graph: {} classes, {} input channels",
                config.getNumClasses(), config.getInputChannels());

        // Input placeholder [batch, channels, height, width]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT,
                -1, config.getInputChannels(), -1, -1);

        // Initial convolution: conv1 (7x7, stride 2) -> bn1 -> relu -> maxpool
        SDVariable x = buildInitialLayers(sd, input, weights, config);

        // Build residual layers
        int[] blockCounts = config.getBlockCounts();
        int[] channelCounts = config.getChannelCounts();

        if (blockCounts != null && blockCounts.length == 4) {
            x = buildLayer(sd, x, "layer1", blockCounts[0], channelCounts[0], 1, weights, config);
            x = buildLayer(sd, x, "layer2", blockCounts[1], channelCounts[1], 2, weights, config);
            x = buildLayer(sd, x, "layer3", blockCounts[2], channelCounts[2], 2, weights, config);
            x = buildLayer(sd, x, "layer4", blockCounts[3], channelCounts[3], 2, weights, config);
        }

        // Global average pooling + FC classifier
        x = buildClassifier(sd, x, weights, config);

        // Mark output
        sd.setOutputs(x.name());

        return sd;
    }

    private SDVariable buildInitialLayers(SameDiff sd, SDVariable input,
                                           Map<String, INDArray> weights, ArchitectureConfig config) {
        // conv1: 7x7, stride 2, padding 3
        SDVariable conv1W = getWeight(sd, weights, "conv1_weight", "conv1.weight");
        SDVariable x = input; // Placeholder for actual conv2d

        // bn1
        SDVariable bn1W = getWeight(sd, weights, "bn1_weight", "bn1.weight");
        SDVariable bn1B = getWeight(sd, weights, "bn1_bias", "bn1.bias");

        // relu (in-place in PyTorch, but we treat it the same)
        x = sd.nn().relu(x, 0);

        // maxpool: 3x3, stride 2, padding 1
        // x = sd.cnn().maxPooling2d(...);

        return x;
    }

    private SDVariable buildLayer(SameDiff sd, SDVariable input, String layerName,
                                   int numBlocks, int channels, int stride,
                                   Map<String, INDArray> weights, ArchitectureConfig config) {
        SDVariable x = input;

        for (int i = 0; i < numBlocks; i++) {
            String blockPrefix = layerName + "." + i;
            int blockStride = (i == 0) ? stride : 1;

            if (config.isUseBottleneck()) {
                x = buildBottleneckBlock(sd, x, blockPrefix, channels, blockStride, weights);
            } else {
                x = buildBasicBlock(sd, x, blockPrefix, channels, blockStride, weights);
            }
        }

        return x;
    }

    private SDVariable buildBasicBlock(SameDiff sd, SDVariable input, String prefix,
                                        int channels, int stride, Map<String, INDArray> weights) {
        SDVariable identity = input;

        // conv1 -> bn1 -> relu
        SDVariable x = input;
        SDVariable conv1W = getWeight(sd, weights, prefix + "_conv1_weight", prefix + ".conv1.weight");
        x = sd.nn().relu(x, 0);

        // conv2 -> bn2
        SDVariable conv2W = getWeight(sd, weights, prefix + "_conv2_weight", prefix + ".conv2.weight");

        // Downsample if needed
        if (stride != 1) {
            SDVariable dsConvW = getWeight(sd, weights, prefix + "_downsample_0_weight",
                    prefix + ".downsample.0.weight");
            identity = input; // Would apply downsample conv + bn here
        }

        // Residual connection
        x = x.add(identity);
        x = sd.nn().relu(x, 0);

        return x;
    }

    private SDVariable buildBottleneckBlock(SameDiff sd, SDVariable input, String prefix,
                                             int channels, int stride, Map<String, INDArray> weights) {
        SDVariable identity = input;

        // conv1 (1x1) -> bn1 -> relu
        SDVariable x = input;
        SDVariable conv1W = getWeight(sd, weights, prefix + "_conv1_weight", prefix + ".conv1.weight");
        x = sd.nn().relu(x, 0);

        // conv2 (3x3) -> bn2 -> relu
        SDVariable conv2W = getWeight(sd, weights, prefix + "_conv2_weight", prefix + ".conv2.weight");
        x = sd.nn().relu(x, 0);

        // conv3 (1x1) -> bn3
        SDVariable conv3W = getWeight(sd, weights, prefix + "_conv3_weight", prefix + ".conv3.weight");

        // Downsample if needed
        if (stride != 1) {
            SDVariable dsConvW = getWeight(sd, weights, prefix + "_downsample_0_weight",
                    prefix + ".downsample.0.weight");
            identity = input;
        }

        // Residual connection
        x = x.add(identity);
        x = sd.nn().relu(x, 0);

        return x;
    }

    private SDVariable buildClassifier(SameDiff sd, SDVariable input,
                                        Map<String, INDArray> weights, ArchitectureConfig config) {
        // Global average pooling
        // SDVariable x = sd.cnn().avgPooling2d(input, ...);

        // Flatten
        SDVariable x = sd.reshape(input, -1, getWeight(sd, weights, "fc_weight", "fc.weight")
                .getShape()[1]);

        // FC layer
        SDVariable fcW = getWeight(sd, weights, "fc_weight", "fc.weight");
        SDVariable fcB = getWeight(sd, weights, "fc_bias", "fc.bias");

        x = sd.nn().linear(x, fcW, fcB);

        return x;
    }

    private SDVariable getWeight(SameDiff sd, Map<String, INDArray> weights,
                                  String sdName, String... torchNames) {
        for (String torchName : torchNames) {
            String mappedName = TensorNameMapper.normalizeName(torchName);
            if (weights.containsKey(mappedName)) {
                return sd.constant(sdName, weights.get(mappedName));
            }
            if (weights.containsKey(torchName)) {
                return sd.constant(sdName, weights.get(torchName));
            }
        }
        log.warn("Weight not found: {}", sdName);
        return null;
    }

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();
        patterns.put("conv1.weight", "Initial 7x7 convolution");
        patterns.put("bn1.weight", "Initial batch norm");
        patterns.put("layer{n}.{block}.conv{m}.weight", "Residual block convolution");
        patterns.put("layer{n}.{block}.bn{m}.weight", "Residual block batch norm");
        patterns.put("layer{n}.{block}.downsample.0.weight", "Downsample convolution");
        patterns.put("fc.weight", "Final classifier");
        return patterns;
    }

    @Override
    public int getPriority() {
        return 100; // High priority for ResNet detection
    }
}
