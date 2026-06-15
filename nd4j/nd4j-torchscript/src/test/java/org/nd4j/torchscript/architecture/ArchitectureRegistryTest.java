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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.torchscript.format.TorchScriptDataType;
import org.nd4j.torchscript.format.TorchScriptFormat;
import org.nd4j.torchscript.format.TorchScriptMetadata;
import org.nd4j.torchscript.format.TorchScriptTensorInfo;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ArchitectureRegistry Test")
class ArchitectureRegistryTest {

    @Test
    @DisplayName("Test built-in architectures are registered")
    void testBuiltInArchitectures() {
        Set<String> architectures = ArchitectureRegistry.getRegisteredArchitectures();

        assertTrue(architectures.contains("resnet"));
        assertTrue(architectures.contains("vgg"));
        assertTrue(architectures.contains("efficientnet"));
        assertTrue(architectures.contains("generic_cnn"));
    }

    @Test
    @DisplayName("Test getArchitecture by name")
    void testGetArchitectureByName() {
        VisionArchitecture resnet = ArchitectureRegistry.getArchitecture("resnet");
        assertNotNull(resnet);
        assertEquals("resnet", resnet.getName());

        VisionArchitecture vgg = ArchitectureRegistry.getArchitecture("vgg");
        assertNotNull(vgg);
        assertEquals("vgg", vgg.getName());
    }

    @Test
    @DisplayName("Test getArchitecture by variant")
    void testGetArchitectureByVariant() {
        VisionArchitecture resnet18 = ArchitectureRegistry.getArchitecture("resnet18");
        assertNotNull(resnet18);
        assertEquals("resnet", resnet18.getName());

        VisionArchitecture resnet50 = ArchitectureRegistry.getArchitecture("resnet50");
        assertNotNull(resnet50);
        assertEquals("resnet", resnet50.getName());

        VisionArchitecture vgg16 = ArchitectureRegistry.getArchitecture("vgg16");
        assertNotNull(vgg16);
        assertEquals("vgg", vgg16.getName());
    }

    @Test
    @DisplayName("Test getArchitecture is case insensitive")
    void testGetArchitectureCaseInsensitive() {
        assertNotNull(ArchitectureRegistry.getArchitecture("RESNET"));
        assertNotNull(ArchitectureRegistry.getArchitecture("ResNet"));
        assertNotNull(ArchitectureRegistry.getArchitecture("resnet"));
    }

    @Test
    @DisplayName("Test getArchitecture returns null for unknown")
    void testGetArchitectureUnknown() {
        assertNull(ArchitectureRegistry.getArchitecture("unknown_arch"));
        assertNull(ArchitectureRegistry.getArchitecture("not_registered"));
        assertNull(ArchitectureRegistry.getArchitecture(null));
    }

    @Test
    @DisplayName("Test hasArchitecture")
    void testHasArchitecture() {
        assertTrue(ArchitectureRegistry.hasArchitecture("resnet"));
        assertTrue(ArchitectureRegistry.hasArchitecture("vgg16"));
        assertTrue(ArchitectureRegistry.hasArchitecture("generic_cnn"));
        assertFalse(ArchitectureRegistry.hasArchitecture("unknown"));
        assertFalse(ArchitectureRegistry.hasArchitecture(null));
    }

    @Test
    @DisplayName("Test detectArchitecture from metadata architecture field")
    void testDetectArchitectureFromMetadata() {
        TorchScriptMetadata metadata = TorchScriptMetadata.builder()
                .format(TorchScriptFormat.SAFETENSORS)
                .tensors(Collections.emptyList())
                .architecture("resnet50")
                .build();

        VisionArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertNotNull(detected);
        assertEquals("resnet", detected.getName());
    }

    @Test
    @DisplayName("Test detectArchitecture from tensor patterns - ResNet")
    void testDetectArchitectureResNet() {
        List<TorchScriptTensorInfo> tensors = Arrays.asList(
                createTensorInfo("layer1.0.conv1.weight", new long[]{64, 64, 3, 3}),
                createTensorInfo("layer1.0.bn1.weight", new long[]{64}),
                createTensorInfo("layer1.0.downsample.0.weight", new long[]{256, 64, 1, 1}),
                createTensorInfo("layer2.0.conv1.weight", new long[]{128, 64, 3, 3}),
                createTensorInfo("fc.weight", new long[]{1000, 2048})
        );

        TorchScriptMetadata metadata = TorchScriptMetadata.builder()
                .format(TorchScriptFormat.SAFETENSORS)
                .tensors(tensors)
                .build();

        VisionArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertEquals("resnet", detected.getName());
    }

    @Test
    @DisplayName("Test detectArchitecture from tensor patterns - VGG")
    void testDetectArchitectureVGG() {
        List<TorchScriptTensorInfo> tensors = Arrays.asList(
                createTensorInfo("features.0.weight", new long[]{64, 3, 3, 3}),
                createTensorInfo("features.2.weight", new long[]{64, 64, 3, 3}),
                createTensorInfo("features.5.weight", new long[]{128, 64, 3, 3}),
                createTensorInfo("classifier.0.weight", new long[]{4096, 25088}),
                createTensorInfo("classifier.6.weight", new long[]{1000, 4096})
        );

        TorchScriptMetadata metadata = TorchScriptMetadata.builder()
                .format(TorchScriptFormat.SAFETENSORS)
                .tensors(tensors)
                .build();

        VisionArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertEquals("vgg", detected.getName());
    }

    @Test
    @DisplayName("Test detectArchitecture from tensor patterns - EfficientNet")
    void testDetectArchitectureEfficientNet() {
        List<TorchScriptTensorInfo> tensors = Arrays.asList(
                createTensorInfo("_conv_stem.weight", new long[]{32, 3, 3, 3}),
                createTensorInfo("_blocks.0._expand_conv.weight", new long[]{32, 32, 1, 1}),
                createTensorInfo("_blocks.0._depthwise_conv.weight", new long[]{32, 1, 3, 3}),
                createTensorInfo("_blocks.0._se_reduce.weight", new long[]{8, 32, 1, 1}),
                createTensorInfo("_fc.weight", new long[]{1000, 1280})
        );

        TorchScriptMetadata metadata = TorchScriptMetadata.builder()
                .format(TorchScriptFormat.SAFETENSORS)
                .tensors(tensors)
                .build();

        VisionArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertEquals("efficientnet", detected.getName());
    }

    @Test
    @DisplayName("Test detectArchitecture falls back to generic")
    void testDetectArchitectureFallback() {
        List<TorchScriptTensorInfo> tensors = Arrays.asList(
                createTensorInfo("random_layer.weight", new long[]{64, 64}),
                createTensorInfo("another_param.bias", new long[]{32})
        );

        TorchScriptMetadata metadata = TorchScriptMetadata.builder()
                .format(TorchScriptFormat.SAFETENSORS)
                .tensors(tensors)
                .build();

        VisionArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertEquals("generic_cnn", detected.getName());
    }

    @Test
    @DisplayName("Test getAllArchitectures returns ordered list")
    void testGetAllArchitectures() {
        List<VisionArchitecture> all = ArchitectureRegistry.getAllArchitectures();

        assertFalse(all.isEmpty());

        // Verify architectures are ordered by priority (highest first)
        for (int i = 0; i < all.size() - 1; i++) {
            assertTrue(all.get(i).getPriority() >= all.get(i + 1).getPriority(),
                    "Architectures should be ordered by priority");
        }

        // Generic CNN should be last (lowest priority)
        assertEquals("generic_cnn", all.get(all.size() - 1).getName());
    }

    @Test
    @DisplayName("Test architecture priorities")
    void testArchitecturePriorities() {
        VisionArchitecture resnet = ArchitectureRegistry.getArchitecture("resnet");
        VisionArchitecture vgg = ArchitectureRegistry.getArchitecture("vgg");
        VisionArchitecture generic = ArchitectureRegistry.getArchitecture("generic_cnn");

        assertTrue(resnet.getPriority() > generic.getPriority());
        assertTrue(vgg.getPriority() > generic.getPriority());
        assertEquals(-1, generic.getPriority()); // Fallback priority
    }

    private TorchScriptTensorInfo createTensorInfo(String name, long[] shape) {
        return TorchScriptTensorInfo.builder()
                .name(name)
                .dataType(TorchScriptDataType.FLOAT32)
                .shape(shape)
                .dataOffset(0)
                .dataSize(calculateSize(shape) * 4)
                .build();
    }

    private long calculateSize(long[] shape) {
        long size = 1;
        for (long dim : shape) {
            size *= dim;
        }
        return size;
    }
}
