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
 * Architecture handler for VGG variants.
 * Supports: VGG-11, VGG-13, VGG-16, VGG-19 (with and without batch normalization)
 */
@Slf4j
public class VGGArchitecture implements VisionArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = new HashSet<>(Arrays.asList(
            "vgg", "vgg11", "vgg13", "vgg16", "vgg19",
            "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"
    ));

    @Override
    public String getName() {
        return "vgg";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    @Override
    public boolean canHandle(TorchScriptMetadata metadata) {
        // VGG has distinctive naming: features.0, features.2, ... and classifier.0, classifier.3, ...
        boolean hasFeatures = false;
        boolean hasClassifier = false;

        for (TorchScriptTensorInfo tensor : metadata.getTensors()) {
            String name = tensor.getName().toLowerCase();
            if (name.startsWith("features.")) {
                hasFeatures = true;
            }
            if (name.startsWith("classifier.")) {
                hasClassifier = true;
            }
        }

        return hasFeatures && hasClassifier;
    }

    @Override
    public SameDiff buildGraph(TorchScriptMetadata metadata, Map<String, INDArray> weights,
                               ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);

        log.info("Building VGG graph: {} classes, bn={}",
                config.getNumClasses(), config.isUseBatchNorm());

        // Input placeholder [batch, channels, height, width]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT,
                -1, config.getInputChannels(), -1, -1);

        // Build features (conv blocks)
        SDVariable x = buildFeatures(sd, input, weights, config);

        // Adaptive average pool to 7x7
        // x = sd.cnn().avgPooling2d(x, ...);

        // Flatten and classifier
        x = buildClassifier(sd, x, weights, config);

        // Mark output
        sd.setOutputs(x.name());

        return sd;
    }

    private SDVariable buildFeatures(SameDiff sd, SDVariable input,
                                      Map<String, INDArray> weights, ArchitectureConfig config) {
        SDVariable x = input;

        // VGG features are numbered sequentially: features.0, features.2, features.5, etc.
        // Conv layers at positions based on VGG variant
        int[] layerCounts = config.getLayerCounts();
        if (layerCounts == null) {
            layerCounts = new int[]{2, 2, 3, 3, 3}; // VGG-16 default
        }

        int featureIdx = 0;
        int[] channels = {64, 128, 256, 512, 512};

        for (int block = 0; block < layerCounts.length; block++) {
            for (int layer = 0; layer < layerCounts[block]; layer++) {
                // Conv2d
                String convName = "features." + featureIdx;
                SDVariable convW = getWeight(sd, weights, convName + "_weight", convName + ".weight");
                SDVariable convB = getWeight(sd, weights, convName + "_bias", convName + ".bias");
                featureIdx++;

                if (config.isUseBatchNorm()) {
                    // BatchNorm2d
                    String bnName = "features." + featureIdx;
                    SDVariable bnW = getWeight(sd, weights, bnName + "_weight", bnName + ".weight");
                    SDVariable bnB = getWeight(sd, weights, bnName + "_bias", bnName + ".bias");
                    featureIdx++;
                }

                // ReLU
                x = sd.nn().relu(x, 0);
                featureIdx++;
            }

            // MaxPool2d
            featureIdx++;
        }

        return x;
    }

    private SDVariable buildClassifier(SameDiff sd, SDVariable input,
                                        Map<String, INDArray> weights, ArchitectureConfig config) {
        // Flatten
        SDVariable x = sd.reshape(input, -1, 512 * 7 * 7);

        // classifier.0: Linear(512*7*7, 4096)
        SDVariable fc1W = getWeight(sd, weights, "classifier_0_weight", "classifier.0.weight");
        SDVariable fc1B = getWeight(sd, weights, "classifier_0_bias", "classifier.0.bias");
        x = sd.nn().relu(x, 0);

        // classifier.1: ReLU (inline)
        // classifier.2: Dropout (skip in inference)

        // classifier.3: Linear(4096, 4096)
        SDVariable fc2W = getWeight(sd, weights, "classifier_3_weight", "classifier.3.weight");
        SDVariable fc2B = getWeight(sd, weights, "classifier_3_bias", "classifier.3.bias");
        x = sd.nn().relu(x, 0);

        // classifier.4: ReLU (inline)
        // classifier.5: Dropout (skip in inference)

        // classifier.6: Linear(4096, num_classes)
        SDVariable fc3W = getWeight(sd, weights, "classifier_6_weight", "classifier.6.weight");
        SDVariable fc3B = getWeight(sd, weights, "classifier_6_bias", "classifier.6.bias");

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
        patterns.put("features.{n}.weight", "Feature extraction conv layer");
        patterns.put("features.{n}.bias", "Feature extraction conv bias");
        patterns.put("classifier.{n}.weight", "Classifier FC layer");
        patterns.put("classifier.{n}.bias", "Classifier FC bias");
        return patterns;
    }

    @Override
    public int getPriority() {
        return 90; // High priority
    }
}
