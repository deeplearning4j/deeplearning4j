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

// TODO: Move to platform-tests/ per project convention — all tests must run from platform-tests

package org.nd4j.torchscript.convert;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ConversionOptions Test")
class ConversionOptionsTest {

    @Test
    @DisplayName("Test default options")
    void testDefaultOptions() {
        ConversionOptions options = ConversionOptions.builder().build();

        assertEquals(DataType.FLOAT, options.getTargetDataType());
        assertFalse(options.isForTraining());
        assertTrue(options.isUseMemoryMapping());
        assertNull(options.getArchitectureOverride());
        assertNotNull(options.getTensorNameMapping());
        assertTrue(options.getTensorNameMapping().isEmpty());
        assertEquals(0, options.getMaxFileSize());
        assertTrue(options.isPreserveInputNames());
        assertEquals(ConversionOptions.DataFormat.NCHW, options.getDataFormat());
        assertTrue(options.isTransformWeights());
        assertFalse(options.isSkipUnsupportedOps());
        assertNull(options.getDefaultInputShape());
    }

    @Test
    @DisplayName("Test forInference factory")
    void testForInference() {
        ConversionOptions options = ConversionOptions.forInference();

        assertFalse(options.isForTraining());
        assertEquals(DataType.FLOAT, options.getTargetDataType());
    }

    @Test
    @DisplayName("Test forTraining factory")
    void testForTraining() {
        ConversionOptions options = ConversionOptions.forTraining();

        assertTrue(options.isForTraining());
        assertEquals(DataType.FLOAT, options.getTargetDataType());
    }

    @Test
    @DisplayName("Test fp16 factory")
    void testFp16() {
        ConversionOptions options = ConversionOptions.fp16();

        assertFalse(options.isForTraining());
        assertEquals(DataType.HALF, options.getTargetDataType());
    }

    @Test
    @DisplayName("Test bf16 factory")
    void testBf16() {
        ConversionOptions options = ConversionOptions.bf16();

        assertFalse(options.isForTraining());
        assertEquals(DataType.BFLOAT16, options.getTargetDataType());
    }

    @Test
    @DisplayName("Test custom target data type")
    void testCustomTargetDataType() {
        ConversionOptions options = ConversionOptions.builder()
                .targetDataType(DataType.DOUBLE)
                .build();

        assertEquals(DataType.DOUBLE, options.getTargetDataType());
    }

    @Test
    @DisplayName("Test architecture override")
    void testArchitectureOverride() {
        ConversionOptions options = ConversionOptions.builder()
                .architectureOverride("resnet50")
                .build();

        assertEquals("resnet50", options.getArchitectureOverride());
    }

    @Test
    @DisplayName("Test tensor name mapping")
    void testTensorNameMapping() {
        Map<String, String> mappings = new HashMap<>();
        mappings.put("layer1.weight", "custom_weight");
        mappings.put("layer1.bias", "custom_bias");

        ConversionOptions options = ConversionOptions.builder()
                .tensorNameMapping(mappings)
                .build();

        assertEquals(2, options.getTensorNameMapping().size());
        assertEquals("custom_weight", options.getTensorNameMapping().get("layer1.weight"));
    }

    @Test
    @DisplayName("Test max file size")
    void testMaxFileSize() {
        ConversionOptions options = ConversionOptions.builder()
                .maxFileSize(1024 * 1024 * 100) // 100MB
                .build();

        assertEquals(104857600, options.getMaxFileSize());
    }

    @Test
    @DisplayName("Test data format options")
    void testDataFormat() {
        ConversionOptions nchwOptions = ConversionOptions.builder()
                .dataFormat(ConversionOptions.DataFormat.NCHW)
                .build();
        assertEquals(ConversionOptions.DataFormat.NCHW, nchwOptions.getDataFormat());

        ConversionOptions nhwcOptions = ConversionOptions.builder()
                .dataFormat(ConversionOptions.DataFormat.NHWC)
                .build();
        assertEquals(ConversionOptions.DataFormat.NHWC, nhwcOptions.getDataFormat());
    }

    @Test
    @DisplayName("Test transform weights flag")
    void testTransformWeights() {
        ConversionOptions options = ConversionOptions.builder()
                .transformWeights(false)
                .build();

        assertFalse(options.isTransformWeights());
    }

    @Test
    @DisplayName("Test skip unsupported ops flag")
    void testSkipUnsupportedOps() {
        ConversionOptions options = ConversionOptions.builder()
                .skipUnsupportedOps(true)
                .build();

        assertTrue(options.isSkipUnsupportedOps());
    }

    @Test
    @DisplayName("Test default input shape")
    void testDefaultInputShape() {
        long[] shape = new long[]{1, 3, 224, 224};

        ConversionOptions options = ConversionOptions.builder()
                .defaultInputShape(shape)
                .build();

        assertArrayEquals(shape, options.getDefaultInputShape());
    }

    @Test
    @DisplayName("Test combined options")
    void testCombinedOptions() {
        Map<String, String> mappings = new HashMap<>();
        mappings.put("old_name", "new_name");

        ConversionOptions options = ConversionOptions.builder()
                .targetDataType(DataType.HALF)
                .forTraining(false)
                .useMemoryMapping(false)
                .architectureOverride("resnet18")
                .tensorNameMapping(mappings)
                .maxFileSize(50 * 1024 * 1024)
                .preserveInputNames(false)
                .dataFormat(ConversionOptions.DataFormat.NHWC)
                .transformWeights(true)
                .skipUnsupportedOps(true)
                .defaultInputShape(new long[]{1, 224, 224, 3})
                .build();

        assertEquals(DataType.HALF, options.getTargetDataType());
        assertFalse(options.isForTraining());
        assertFalse(options.isUseMemoryMapping());
        assertEquals("resnet18", options.getArchitectureOverride());
        assertEquals(1, options.getTensorNameMapping().size());
        assertEquals(52428800, options.getMaxFileSize());
        assertFalse(options.isPreserveInputNames());
        assertEquals(ConversionOptions.DataFormat.NHWC, options.getDataFormat());
        assertTrue(options.isTransformWeights());
        assertTrue(options.isSkipUnsupportedOps());
        assertArrayEquals(new long[]{1, 224, 224, 3}, options.getDefaultInputShape());
    }
}
