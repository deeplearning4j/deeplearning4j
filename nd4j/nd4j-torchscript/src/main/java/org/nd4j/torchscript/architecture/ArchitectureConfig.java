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

import lombok.Builder;
import lombok.Data;
import org.nd4j.torchscript.convert.ConversionOptions;
import org.nd4j.torchscript.format.TorchScriptMetadata;

/**
 * Configuration holder for CNN architecture parameters.
 */
@Data
@Builder
public class ArchitectureConfig {

    /**
     * Number of input channels (e.g., 3 for RGB images)
     */
    @Builder.Default
    private int inputChannels = 3;

    /**
     * Input image size [height, width]
     */
    @Builder.Default
    private int[] inputSize = new int[]{224, 224};

    /**
     * Number of output classes for classification
     */
    @Builder.Default
    private int numClasses = 1000;

    /**
     * Data format (NCHW or NHWC)
     */
    @Builder.Default
    private ConversionOptions.DataFormat dataFormat = ConversionOptions.DataFormat.NCHW;

    // ResNet-specific configuration

    /**
     * Number of blocks per layer (e.g., [3, 4, 6, 3] for ResNet-50)
     */
    private int[] blockCounts;

    /**
     * Channel counts per layer (e.g., [64, 128, 256, 512])
     */
    @Builder.Default
    private int[] channelCounts = new int[]{64, 128, 256, 512};

    /**
     * Whether to use bottleneck blocks (ResNet-50+) vs basic blocks (ResNet-18/34)
     */
    @Builder.Default
    private boolean useBottleneck = false;

    // VGG-specific configuration

    /**
     * Layer configuration for VGG (number of conv layers per block)
     */
    private int[] layerCounts;

    /**
     * Whether VGG uses batch normalization
     */
    @Builder.Default
    private boolean useBatchNorm = false;

    // EfficientNet-specific configuration

    /**
     * Width multiplier for EfficientNet
     */
    @Builder.Default
    private double widthMultiplier = 1.0;

    /**
     * Depth multiplier for EfficientNet
     */
    @Builder.Default
    private double depthMultiplier = 1.0;

    /**
     * Resolution multiplier for EfficientNet
     */
    @Builder.Default
    private double resolutionMultiplier = 1.0;

    /**
     * Dropout rate
     */
    @Builder.Default
    private double dropoutRate = 0.2;

    /**
     * Create configuration from metadata.
     *
     * @param metadata the model metadata
     * @return architecture config
     */
    public static ArchitectureConfig fromMetadata(TorchScriptMetadata metadata) {
        ArchitectureConfigBuilder builder = ArchitectureConfig.builder()
                .inputChannels(metadata.getInputChannels())
                .numClasses(metadata.getNumClasses());

        // Infer configuration based on architecture
        String arch = metadata.getArchitecture();
        if (arch != null) {
            if (arch.contains("resnet")) {
                configureResNet(builder, arch);
            } else if (arch.contains("vgg")) {
                configureVGG(builder, arch);
            } else if (arch.contains("efficientnet")) {
                configureEfficientNet(builder, arch);
            }
        }

        return builder.build();
    }

    private static void configureResNet(ArchitectureConfigBuilder builder, String arch) {
        if (arch.contains("18")) {
            builder.blockCounts(new int[]{2, 2, 2, 2}).useBottleneck(false);
        } else if (arch.contains("34")) {
            builder.blockCounts(new int[]{3, 4, 6, 3}).useBottleneck(false);
        } else if (arch.contains("50")) {
            builder.blockCounts(new int[]{3, 4, 6, 3}).useBottleneck(true);
        } else if (arch.contains("101")) {
            builder.blockCounts(new int[]{3, 4, 23, 3}).useBottleneck(true);
        } else if (arch.contains("152")) {
            builder.blockCounts(new int[]{3, 8, 36, 3}).useBottleneck(true);
        }
    }

    private static void configureVGG(ArchitectureConfigBuilder builder, String arch) {
        builder.useBatchNorm(arch.contains("bn"));

        if (arch.contains("11")) {
            builder.layerCounts(new int[]{1, 1, 2, 2, 2});
        } else if (arch.contains("13")) {
            builder.layerCounts(new int[]{2, 2, 2, 2, 2});
        } else if (arch.contains("16")) {
            builder.layerCounts(new int[]{2, 2, 3, 3, 3});
        } else if (arch.contains("19")) {
            builder.layerCounts(new int[]{2, 2, 4, 4, 4});
        }
    }

    private static void configureEfficientNet(ArchitectureConfigBuilder builder, String arch) {
        // EfficientNet compound scaling coefficients
        if (arch.contains("b0")) {
            builder.widthMultiplier(1.0).depthMultiplier(1.0).resolutionMultiplier(1.0).dropoutRate(0.2);
        } else if (arch.contains("b1")) {
            builder.widthMultiplier(1.0).depthMultiplier(1.1).resolutionMultiplier(1.0).dropoutRate(0.2);
        } else if (arch.contains("b2")) {
            builder.widthMultiplier(1.1).depthMultiplier(1.2).resolutionMultiplier(1.0).dropoutRate(0.3);
        } else if (arch.contains("b3")) {
            builder.widthMultiplier(1.2).depthMultiplier(1.4).resolutionMultiplier(1.0).dropoutRate(0.3);
        } else if (arch.contains("b4")) {
            builder.widthMultiplier(1.4).depthMultiplier(1.8).resolutionMultiplier(1.0).dropoutRate(0.4);
        } else if (arch.contains("b5")) {
            builder.widthMultiplier(1.6).depthMultiplier(2.2).resolutionMultiplier(1.0).dropoutRate(0.4);
        } else if (arch.contains("b6")) {
            builder.widthMultiplier(1.8).depthMultiplier(2.6).resolutionMultiplier(1.0).dropoutRate(0.5);
        } else if (arch.contains("b7")) {
            builder.widthMultiplier(2.0).depthMultiplier(3.1).resolutionMultiplier(1.0).dropoutRate(0.5);
        }
    }
}
