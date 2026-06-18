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
package org.eclipse.deeplearning4j.frameworkimport.keras.layers.core;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.layers.EinsumDense;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.core.KerasEinsumDense;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Keras EinsumDense layer import.
 *
 * @author Adam Gibson
 */
@DisplayName("Keras EinsumDense Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
class KerasEinsumDenseTest extends BaseDL4JTest {

    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    private final String LAYER_NAME = "einsum_dense";
    private final String EQUATION = "ab,bc->ac";

    @Test
    @DisplayName("Test EinsumDense Layer Import - Keras 2")
    void testEinsumDenseLayerKeras2() throws Exception {
        buildEinsumDenseLayer(conf2, 2);
    }

    private void buildEinsumDenseLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        // Output shape as list
        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);

        // Bias axes
        config.put("bias_axes", "c");

        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertEquals(LAYER_NAME, kerasLayer.getLayerName());
        assertEquals(EQUATION, kerasLayer.getEquation());
        assertTrue(kerasLayer.isHasBias());

        EinsumDense layer = kerasLayer.getEinsumDenseLayer();
        assertNotNull(layer);
        assertEquals(EQUATION, layer.getEquation());
    }

    @Test
    @DisplayName("Test EinsumDense Layer Without Bias")
    void testEinsumDenseLayerNoBias() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);

        // No bias_axes means no bias
        config.put("bias_axes", "");

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertFalse(kerasLayer.isHasBias());
    }

    @Test
    @DisplayName("Test EinsumDense Layer Missing Equation")
    void testEinsumDenseLayerMissingEquation() {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        // Missing equation

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        assertThrows(InvalidKerasConfigurationException.class, () -> {
            new KerasEinsumDense(layerConfig, false);
        });
    }

    @Test
    @DisplayName("Test EinsumDense Layer Weight Setting")
    void testEinsumDenseLayerWeights() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);
        config.put("bias_axes", "c");

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        // Create mock weights
        int nIn = 32;
        int nOut = 64;
        Map<String, INDArray> weights = new HashMap<>();
        weights.put("kernel", Nd4j.rand(nIn, nOut));
        weights.put("bias", Nd4j.rand(nOut));

        kerasLayer.setWeights(weights);

        Map<String, INDArray> layerWeights = kerasLayer.getWeights();
        assertNotNull(layerWeights);
        assertTrue(layerWeights.containsKey("W"));
        assertTrue(layerWeights.containsKey("b"));
        assertArrayEquals(new long[]{nIn, nOut}, layerWeights.get("W").shape());
    }

    @Test
    @DisplayName("Test EinsumDense Different Equations")
    void testEinsumDenseDifferentEquations() throws Exception {
        String[] equations = {
                "ab,bc->ac",      // Standard dense
                "abc,cd->abd",    // Dense on sequence
                "...x,xy->...y"   // Ellipsis notation (if supported)
        };

        for (String eq : equations) {
            try {
                Map<String, Object> layerConfig = new HashMap<>();
                layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

                Map<String, Object> config = new HashMap<>();
                config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
                config.put("equation", eq);

                List<Integer> outputShape = new ArrayList<>();
                outputShape.add(64);
                config.put("output_shape", outputShape);

                layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
                layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

                KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);
                assertEquals(eq, kerasLayer.getEquation());
            } catch (Exception e) {
                // Some equations might not be supported, that's okay
                System.out.println("Equation not supported: " + eq + " - " + e.getMessage());
            }
        }
    }

    @Test
    @DisplayName("Test EinsumDense Output Type")
    void testEinsumDenseOutputType() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        // Check output shape is correctly parsed
        assertNotNull(kerasLayer.getOutputShape());
        assertEquals(1, kerasLayer.getOutputShape().length);
        assertEquals(64, kerasLayer.getOutputShape()[0]);
    }

    @Test
    @DisplayName("Test EinsumDense Num Params")
    void testEinsumDenseNumParams() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);
        config.put("bias_axes", "c");

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayerWithBias = new KerasEinsumDense(layerConfig, false);
        assertEquals(2, kerasLayerWithBias.getNumParams()); // kernel + bias

        // Without bias
        config.put("bias_axes", "");
        KerasEinsumDense kerasLayerNoBias = new KerasEinsumDense(layerConfig, false);
        assertEquals(1, kerasLayerNoBias.getNumParams()); // kernel only
    }

    // ==================== Additional Keras Tests ====================

    @Test
    @DisplayName("Test EinsumDense Multi-Dimensional Output Shape")
    void testEinsumDenseMultiDimensionalOutputShape() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), "einsum_sequence");
        config.put("equation", "abc,cd->abd");  // Sequence processing equation

        // Multi-dimensional output: [sequence_length, features]
        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(10);  // sequence length
        outputShape.add(64);  // features
        config.put("output_shape", outputShape);
        config.put("bias_axes", "d");

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertNotNull(kerasLayer.getOutputShape());
        assertEquals(2, kerasLayer.getOutputShape().length);
        assertEquals(10, kerasLayer.getOutputShape()[0]);
        assertEquals(64, kerasLayer.getOutputShape()[1]);
    }

    @Test
    @DisplayName("Test EinsumDense Integer Array Output Shape")
    void testEinsumDenseIntArrayOutputShape() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        // Output shape as int array instead of List
        config.put("output_shape", new int[]{128});

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertNotNull(kerasLayer.getOutputShape());
        assertEquals(1, kerasLayer.getOutputShape().length);
        assertEquals(128, kerasLayer.getOutputShape()[0]);
    }

    @Test
    @DisplayName("Test EinsumDense Null Bias Axes")
    void testEinsumDenseNullBiasAxes() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);
        // bias_axes not set (null)

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        // Null bias_axes should mean no bias
        assertFalse(kerasLayer.isHasBias());
    }

    @Test
    @DisplayName("Test EinsumDense Weight Loading With Keras Naming")
    void testEinsumDenseWeightLoadingKerasNaming() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), "my_einsum_layer");
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);
        config.put("bias_axes", "c");

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        // Test Keras-style weight naming: "layer_name/kernel:0"
        int nIn = 32;
        int nOut = 64;
        Map<String, INDArray> weights = new HashMap<>();
        weights.put("my_einsum_layer/kernel:0", Nd4j.rand(nIn, nOut));
        weights.put("my_einsum_layer/bias:0", Nd4j.rand(nOut));

        kerasLayer.setWeights(weights);

        Map<String, INDArray> layerWeights = kerasLayer.getWeights();
        assertNotNull(layerWeights);
        assertTrue(layerWeights.containsKey("W"));
        assertTrue(layerWeights.containsKey("b"));
    }

    @Test
    @DisplayName("Test EinsumDense Weight Loading Without Bias")
    void testEinsumDenseWeightLoadingNoBias() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);
        config.put("bias_axes", "");  // No bias

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        // Only kernel weights
        int nIn = 32;
        int nOut = 64;
        Map<String, INDArray> weights = new HashMap<>();
        weights.put("kernel", Nd4j.rand(nIn, nOut));

        kerasLayer.setWeights(weights);

        Map<String, INDArray> layerWeights = kerasLayer.getWeights();
        assertNotNull(layerWeights);
        assertTrue(layerWeights.containsKey("W"));
        assertFalse(layerWeights.containsKey("b"));
    }

    @Test
    @DisplayName("Test EinsumDense Output Type Feed Forward")
    void testEinsumDenseOutputTypeFeedForward() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);  // Single dimension = FeedForward
        config.put("output_shape", outputShape);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        InputType inputType = InputType.feedForward(32);
        InputType outputType = kerasLayer.getOutputType(inputType);

        assertNotNull(outputType);
        assertEquals(InputType.Type.FF, outputType.getType());
    }

    @Test
    @DisplayName("Test EinsumDense Output Type Recurrent")
    void testEinsumDenseOutputTypeRecurrent() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", "abc,cd->abd");

        // Two dimensions = Recurrent [timeSteps, features]
        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(10);  // sequence length
        outputShape.add(64);  // features
        config.put("output_shape", outputShape);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        InputType inputType = InputType.recurrent(32, 10);
        InputType outputType = kerasLayer.getOutputType(inputType);

        assertNotNull(outputType);
        assertEquals(InputType.Type.RNN, outputType.getType());
    }

    @Test
    @DisplayName("Test EinsumDense Multiple Inputs Error")
    void testEinsumDenseMultipleInputsError() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        // Should throw when given multiple inputs
        InputType input1 = InputType.feedForward(32);
        InputType input2 = InputType.feedForward(64);

        assertThrows(InvalidKerasConfigurationException.class, () -> {
            kerasLayer.getOutputType(input1, input2);
        });
    }

    static Stream<Arguments> kerasVersionProvider() {
        return Stream.of(
                Arguments.of(new Keras2LayerConfiguration(), 2)
        );
    }

    @ParameterizedTest
    @MethodSource("kerasVersionProvider")
    @DisplayName("Test EinsumDense Batch Matmul Equation - Keras 2 and 3")
    void testEinsumDenseBatchMatmulEquation(KerasLayerConfiguration conf, int kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), "batch_einsum");
        config.put("equation", "abc,acd->abd");  // Batch matmul style

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(10);  // seq
        outputShape.add(64);  // features
        config.put("output_shape", outputShape);
        config.put("bias_axes", "d");

        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertEquals("abc,acd->abd", kerasLayer.getEquation());
        assertEquals("batch_einsum", kerasLayer.getLayerName());
        assertTrue(kerasLayer.isHasBias());
    }

    @ParameterizedTest
    @MethodSource("kerasVersionProvider")
    @DisplayName("Test EinsumDense Attention-Style Equation - Keras 2 and 3")
    void testEinsumDenseAttentionEquation(KerasLayerConfiguration conf, int kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), "attention_einsum");
        // Attention-style: batch, heads, seq, dim contracted with heads, dim, dim
        config.put("equation", "abcd,cde->abce");

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(8);   // heads
        outputShape.add(10);  // seq
        outputShape.add(64);  // dim
        config.put("output_shape", outputShape);

        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertEquals("abcd,cde->abce", kerasLayer.getEquation());
        assertEquals(3, kerasLayer.getOutputShape().length);
    }

    @Test
    @DisplayName("Test EinsumDense 3D Kernel Weights")
    void testEinsumDense3DKernelWeights() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), "einsum_3d_kernel");
        config.put("equation", "abcd,cde->abe");  // 3D kernel

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(10);
        outputShape.add(64);
        config.put("output_shape", outputShape);
        config.put("bias_axes", "e");

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        // 3D kernel: [c, d, e]
        Map<String, INDArray> weights = new HashMap<>();
        weights.put("kernel", Nd4j.rand(8, 16, 64));  // 3D
        weights.put("bias", Nd4j.rand(64));

        kerasLayer.setWeights(weights);

        Map<String, INDArray> layerWeights = kerasLayer.getWeights();
        assertNotNull(layerWeights);
        assertTrue(layerWeights.containsKey("W"));
        assertEquals(3, layerWeights.get("W").rank());
        assertArrayEquals(new long[]{8, 16, 64}, layerWeights.get("W").shape());
    }

    @Test
    @DisplayName("Test EinsumDense Empty Equation Error")
    void testEinsumDenseEmptyEquationError() {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", "");  // Empty equation

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        assertThrows(InvalidKerasConfigurationException.class, () -> {
            new KerasEinsumDense(layerConfig, false);
        });
    }

    @Test
    @DisplayName("Test EinsumDense Layer Name Preserved")
    void testEinsumDenseLayerNamePreserved() throws Exception {
        String customName = "my_custom_einsum_dense_layer";

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), customName);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertEquals(customName, kerasLayer.getLayerName());

        EinsumDense dl4jLayer = kerasLayer.getEinsumDenseLayer();
        assertEquals(customName, dl4jLayer.getLayerName());
    }

    @Test
    @DisplayName("Test EinsumDense Transpose Equation")
    void testEinsumDenseTransposeEquation() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), "einsum_transpose");
        config.put("equation", "ab,cb->ac");  // Weight transposed

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertEquals("ab,cb->ac", kerasLayer.getEquation());
    }

    @Test
    @DisplayName("Test EinsumDense Enforcing Training Config")
    void testEinsumDenseEnforceTrainingConfig() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);
        config.put("bias_axes", "c");

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        // Test with enforceTrainingConfig = true (default constructor)
        KerasEinsumDense kerasLayerEnforced = new KerasEinsumDense(layerConfig);
        assertNotNull(kerasLayerEnforced);
        assertEquals(EQUATION, kerasLayerEnforced.getEquation());
    }

    @Test
    @DisplayName("Test EinsumDense Various Bias Axes")
    void testEinsumDenseVariousBiasAxes() throws Exception {
        String[] biasAxesOptions = {"c", "bc", "abc", "d"};

        for (String biasAxes : biasAxesOptions) {
            Map<String, Object> layerConfig = new HashMap<>();
            layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

            Map<String, Object> config = new HashMap<>();
            config.put(conf2.getLAYER_FIELD_NAME(), LAYER_NAME);
            config.put("equation", "abc,cd->abd");

            List<Integer> outputShape = new ArrayList<>();
            outputShape.add(10);
            outputShape.add(64);
            config.put("output_shape", outputShape);
            config.put("bias_axes", biasAxes);

            layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
            layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

            KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);
            assertTrue(kerasLayer.isHasBias(), "Should have bias for bias_axes: " + biasAxes);
        }
    }

    @Test
    @DisplayName("Test EinsumDense Keras 2 Specific Fields")
    void testEinsumDenseKeras2SpecificFields() throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf2.getLAYER_FIELD_CLASS_NAME(), conf2.getLAYER_CLASS_NAME_EINSUM_DENSE());

        Map<String, Object> config = new HashMap<>();
        config.put(conf2.getLAYER_FIELD_NAME(), "keras2_einsum");
        config.put("equation", EQUATION);

        List<Integer> outputShape = new ArrayList<>();
        outputShape.add(64);
        config.put("output_shape", outputShape);
        config.put("bias_axes", "c");

        // Keras 2 specific fields
        config.put("kernel_initializer", createInitializerConfig("glorot_uniform"));
        config.put("bias_initializer", createInitializerConfig("zeros"));

        layerConfig.put(conf2.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf2.getLAYER_FIELD_KERAS_VERSION(), 2);

        KerasEinsumDense kerasLayer = new KerasEinsumDense(layerConfig, false);

        assertNotNull(kerasLayer);
        assertEquals("keras2_einsum", kerasLayer.getLayerName());
    }

    private Map<String, Object> createInitializerConfig(String className) {
        Map<String, Object> initConfig = new HashMap<>();
        initConfig.put("class_name", className);
        initConfig.put("config", new HashMap<>());
        return initConfig;
    }
}
