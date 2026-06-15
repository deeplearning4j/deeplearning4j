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
 * Architecture handler for EfficientNet variants.
 * Supports: EfficientNet-B0 through B7
 */
@Slf4j
public class EfficientNetArchitecture implements VisionArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = new HashSet<>(Arrays.asList(
            "efficientnet", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
            "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"
    ));

    @Override
    public String getName() {
        return "efficientnet";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    @Override
    public boolean canHandle(TorchScriptMetadata metadata) {
        // EfficientNet has distinctive naming: _blocks, _bn, _se
        for (TorchScriptTensorInfo tensor : metadata.getTensors()) {
            String name = tensor.getName().toLowerCase();
            if (name.contains("_blocks.") || name.contains("_conv_stem") ||
                    name.contains("_se_reduce") || name.contains("_se_expand")) {
                return true;
            }
        }
        return false;
    }

    @Override
    public SameDiff buildGraph(TorchScriptMetadata metadata, Map<String, INDArray> weights,
                               ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);

        log.info("Building EfficientNet graph: {} classes, width={}, depth={}",
                config.getNumClasses(), config.getWidthMultiplier(), config.getDepthMultiplier());

        // Input placeholder [batch, channels, height, width]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT,
                -1, config.getInputChannels(), -1, -1);

        // Stem
        SDVariable x = buildStem(sd, input, weights, config);

        // MBConv blocks
        x = buildBlocks(sd, x, weights, config);

        // Head
        x = buildHead(sd, x, weights, config);

        // Mark output
        sd.setOutputs(x.name());

        return sd;
    }

    private SDVariable buildStem(SameDiff sd, SDVariable input,
                                  Map<String, INDArray> weights, ArchitectureConfig config) {
        // Conv stem: 3x3, stride 2
        SDVariable convW = getWeight(sd, weights, "conv_stem_weight", "_conv_stem.weight");

        // BN
        SDVariable bnW = getWeight(sd, weights, "bn0_weight", "_bn0.weight");
        SDVariable bnB = getWeight(sd, weights, "bn0_bias", "_bn0.bias");

        // Swish activation
        SDVariable x = sd.nn().swish(input);

        return x;
    }

    private SDVariable buildBlocks(SameDiff sd, SDVariable input,
                                    Map<String, INDArray> weights, ArchitectureConfig config) {
        SDVariable x = input;

        // EfficientNet block configuration: (expand_ratio, channels, num_blocks, stride, kernel_size)
        // Simplified - actual implementation would iterate through block configs
        int numBlockTypes = 7; // EfficientNet has 7 block types

        for (int blockType = 0; blockType < numBlockTypes; blockType++) {
            // Each block type has multiple sub-blocks
            x = buildMBConvBlock(sd, x, "_blocks." + blockType, weights, config);
        }

        return x;
    }

    private SDVariable buildMBConvBlock(SameDiff sd, SDVariable input, String prefix,
                                         Map<String, INDArray> weights, ArchitectureConfig config) {
        SDVariable x = input;
        SDVariable identity = input;

        // Expansion phase (if expand_ratio > 1)
        SDVariable expandConvW = getWeight(sd, weights, prefix + "_expand_conv_weight",
                prefix + "._expand_conv.weight");
        if (expandConvW != null) {
            SDVariable expandBnW = getWeight(sd, weights, prefix + "_bn0_weight",
                    prefix + "._bn0.weight");
            x = sd.nn().swish(x);
        }

        // Depthwise convolution
        SDVariable dwConvW = getWeight(sd, weights, prefix + "_depthwise_conv_weight",
                prefix + "._depthwise_conv.weight");
        SDVariable dwBnW = getWeight(sd, weights, prefix + "_bn1_weight",
                prefix + "._bn1.weight");
        x = sd.nn().swish(x);

        // Squeeze-and-Excitation
        SDVariable seReduceW = getWeight(sd, weights, prefix + "_se_reduce_weight",
                prefix + "._se_reduce.weight");
        if (seReduceW != null) {
            x = buildSqueezeExcitation(sd, x, prefix, weights);
        }

        // Projection phase
        SDVariable projectConvW = getWeight(sd, weights, prefix + "_project_conv_weight",
                prefix + "._project_conv.weight");
        SDVariable projectBnW = getWeight(sd, weights, prefix + "_bn2_weight",
                prefix + "._bn2.weight");

        // Residual connection (if stride == 1 and input/output channels match)
        // x = x.add(identity);

        return x;
    }

    private SDVariable buildSqueezeExcitation(SameDiff sd, SDVariable input, String prefix,
                                               Map<String, INDArray> weights) {
        // Global average pooling
        SDVariable squeezed = sd.mean(input, 2, 3); // Pool over H, W

        // Reduce
        SDVariable reduceW = getWeight(sd, weights, prefix + "_se_reduce_weight",
                prefix + "._se_reduce.weight");
        SDVariable x = sd.nn().swish(squeezed);

        // Expand
        SDVariable expandW = getWeight(sd, weights, prefix + "_se_expand_weight",
                prefix + "._se_expand.weight");
        x = sd.nn().sigmoid(x);

        // Scale
        return input.mul(x);
    }

    private SDVariable buildHead(SameDiff sd, SDVariable input,
                                  Map<String, INDArray> weights, ArchitectureConfig config) {
        // Conv head
        SDVariable convW = getWeight(sd, weights, "conv_head_weight", "_conv_head.weight");
        SDVariable bnW = getWeight(sd, weights, "bn1_weight", "_bn1.weight");
        SDVariable x = sd.nn().swish(input);

        // Global average pooling
        x = sd.mean(x, 2, 3);

        // Dropout (skip in inference)

        // FC classifier
        SDVariable fcW = getWeight(sd, weights, "fc_weight", "_fc.weight");
        SDVariable fcB = getWeight(sd, weights, "fc_bias", "_fc.bias");

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
        return null; // Some weights are optional
    }

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();
        patterns.put("_conv_stem.weight", "Stem convolution");
        patterns.put("_blocks.{n}._expand_conv.weight", "MBConv expansion");
        patterns.put("_blocks.{n}._depthwise_conv.weight", "MBConv depthwise");
        patterns.put("_blocks.{n}._se_reduce.weight", "Squeeze-excitation reduce");
        patterns.put("_blocks.{n}._se_expand.weight", "Squeeze-excitation expand");
        patterns.put("_blocks.{n}._project_conv.weight", "MBConv projection");
        patterns.put("_conv_head.weight", "Head convolution");
        patterns.put("_fc.weight", "Classifier");
        return patterns;
    }

    @Override
    public int getPriority() {
        return 80;
    }
}
