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

package org.nd4j.torchscript.format;

import lombok.Builder;
import lombok.Data;
import org.nd4j.torchscript.ir.TorchScriptGraph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * High-level metadata for a TorchScript or Safetensors model.
 */
@Data
@Builder
public class TorchScriptMetadata {

    /**
     * File format of the source model
     */
    private TorchScriptFormat format;

    /**
     * Detected architecture (e.g., "resnet", "vgg", "efficientnet")
     */
    private String architecture;

    /**
     * Model name if available
     */
    private String modelName;

    /**
     * Number of convolutional layers detected
     */
    @Builder.Default
    private int numConvLayers = 0;

    /**
     * Number of linear/dense layers detected
     */
    @Builder.Default
    private int numLinearLayers = 0;

    /**
     * Input channels (for CNN models)
     */
    @Builder.Default
    private int inputChannels = 3;

    /**
     * Number of output classes (for classification models)
     */
    @Builder.Default
    private int numClasses = 0;

    /**
     * List of tensor information
     */
    @Builder.Default
    private List<TorchScriptTensorInfo> tensors = new ArrayList<>();

    /**
     * Raw metadata from file (format-specific)
     */
    @Builder.Default
    private Map<String, Object> rawMetadata = new HashMap<>();

    /**
     * Parsed computation graph (for TorchScript files)
     */
    private TorchScriptGraph graph;

    /**
     * Create metadata from a parsed TorchScript graph.
     *
     * @param graph   the parsed graph
     * @param tensors list of tensor info
     * @return metadata instance
     */
    public static TorchScriptMetadata fromTorchScript(TorchScriptGraph graph, List<TorchScriptTensorInfo> tensors) {
        TorchScriptMetadataBuilder builder = TorchScriptMetadata.builder()
                .format(TorchScriptFormat.TORCHSCRIPT_PT)
                .graph(graph)
                .tensors(tensors);

        // Analyze tensors to infer architecture
        analyzeArchitecture(builder, tensors);

        return builder.build();
    }

    /**
     * Create metadata from a Safetensors header.
     *
     * @param header  the parsed header
     * @param tensors list of tensor info
     * @return metadata instance
     */
    public static TorchScriptMetadata fromSafetensors(SafetensorsHeader header, List<TorchScriptTensorInfo> tensors) {
        TorchScriptMetadataBuilder builder = TorchScriptMetadata.builder()
                .format(TorchScriptFormat.SAFETENSORS)
                .tensors(tensors)
                .rawMetadata(header.getMetadata());

        // Analyze tensors to infer architecture
        analyzeArchitecture(builder, tensors);

        return builder.build();
    }

    private static void analyzeArchitecture(TorchScriptMetadataBuilder builder, List<TorchScriptTensorInfo> tensors) {
        int convLayers = 0;
        int linearLayers = 0;
        int inputChannels = 3;
        int numClasses = 0;
        String architecture = "unknown";

        for (TorchScriptTensorInfo tensor : tensors) {
            String name = tensor.getName().toLowerCase();
            long[] shape = tensor.getShape();

            // Count layer types
            if (name.contains("conv") && name.endsWith(".weight")) {
                convLayers++;
                // First conv layer input channels
                if (convLayers == 1 && shape.length == 4) {
                    inputChannels = (int) shape[1]; // PyTorch: [out, in, H, W]
                }
            }
            if ((name.contains("fc") || name.contains("linear") || name.contains("classifier"))
                    && name.endsWith(".weight")) {
                linearLayers++;
                // Last linear layer for num classes
                if (shape.length == 2) {
                    numClasses = (int) shape[0]; // PyTorch: [out, in]
                }
            }

            // Detect architecture from naming patterns
            if (architecture.equals("unknown")) {
                if (name.contains("layer1") || name.contains("layer2") ||
                        name.contains("downsample") || name.contains("bottleneck")) {
                    architecture = "resnet";
                } else if (name.contains("features.") && name.contains("classifier.")) {
                    if (convLayers <= 19) {
                        architecture = "vgg";
                    }
                } else if (name.contains("_blocks.") || name.contains("_bn") ||
                        name.contains("_se_")) {
                    architecture = "efficientnet";
                }
            }
        }

        builder.numConvLayers(convLayers)
                .numLinearLayers(linearLayers)
                .inputChannels(inputChannels)
                .numClasses(numClasses)
                .architecture(architecture);
    }

    /**
     * Get the total size of all tensors in bytes.
     *
     * @return total size
     */
    public long getTotalModelSize() {
        long total = 0;
        for (TorchScriptTensorInfo tensor : tensors) {
            total += tensor.getDataSize();
        }
        return total;
    }

    /**
     * Get the total number of parameters.
     *
     * @return total parameters
     */
    public long getTotalParameters() {
        long total = 0;
        for (TorchScriptTensorInfo tensor : tensors) {
            if (!tensor.isBuffer()) {
                total += tensor.getNumElements();
            }
        }
        return total;
    }

    /**
     * Get the total number of trainable parameters.
     *
     * @return trainable parameters
     */
    public long getTrainableParameters() {
        long total = 0;
        for (TorchScriptTensorInfo tensor : tensors) {
            if (tensor.isRequiresGrad()) {
                total += tensor.getNumElements();
            }
        }
        return total;
    }

    /**
     * Get tensor info by name.
     *
     * @param name tensor name
     * @return tensor info or null
     */
    public TorchScriptTensorInfo getTensor(String name) {
        for (TorchScriptTensorInfo tensor : tensors) {
            if (tensor.getName().equals(name)) {
                return tensor;
            }
        }
        return null;
    }

    /**
     * Check if this model has a computation graph.
     *
     * @return true if graph is available
     */
    public boolean hasGraph() {
        return graph != null;
    }
}
