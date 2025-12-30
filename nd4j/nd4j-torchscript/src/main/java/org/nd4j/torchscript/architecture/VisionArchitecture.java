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

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.torchscript.convert.ConversionOptions;
import org.nd4j.torchscript.format.TorchScriptMetadata;

import java.util.Map;
import java.util.Set;

/**
 * Interface for vision model architecture handlers.
 * Each implementation handles a specific CNN architecture (ResNet, VGG, EfficientNet, etc.)
 * and knows how to build the corresponding SameDiff graph.
 */
public interface VisionArchitecture {

    /**
     * Get the primary name of this architecture (e.g., "resnet", "vgg", "efficientnet")
     *
     * @return architecture name
     */
    String getName();

    /**
     * Get all supported variant names for this architecture.
     * For example, ResNet architecture might support: ["resnet18", "resnet34", "resnet50", ...]
     *
     * @return set of supported variant names
     */
    Set<String> getSupportedVariants();

    /**
     * Check if this architecture can handle the given model metadata.
     *
     * @param metadata the model metadata
     * @return true if this architecture can handle the model
     */
    boolean canHandle(TorchScriptMetadata metadata);

    /**
     * Build the SameDiff graph for this architecture.
     *
     * @param metadata the model metadata
     * @param weights  map of tensor names to weight arrays
     * @param options  conversion options
     * @return the constructed SameDiff graph
     */
    SameDiff buildGraph(TorchScriptMetadata metadata, Map<String, INDArray> weights, ConversionOptions options);

    /**
     * Get the expected tensor name patterns for this architecture.
     * Used for mapping PyTorch tensor names to SameDiff variable names.
     *
     * @return map of pattern names to descriptions
     */
    Map<String, String> getTensorNamePatterns();

    /**
     * Get architecture-specific configuration from metadata.
     *
     * @param metadata the model metadata
     * @return architecture configuration
     */
    default ArchitectureConfig getConfig(TorchScriptMetadata metadata) {
        return ArchitectureConfig.fromMetadata(metadata);
    }

    /**
     * Get the priority of this architecture for auto-detection.
     * Higher priority architectures are checked first.
     *
     * @return priority value (higher = checked first)
     */
    default int getPriority() {
        return 0;
    }
}
