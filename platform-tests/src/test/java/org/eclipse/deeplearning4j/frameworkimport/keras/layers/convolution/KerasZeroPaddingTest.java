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
package org.eclipse.deeplearning4j.frameworkimport.keras.layers.convolution;

import org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPadding3DLayer;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasZeroPadding1D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasZeroPadding2D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasZeroPadding3D;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Parameterized test covering ZeroPadding1D, ZeroPadding2D, and ZeroPadding3D Keras layer import.
 *
 * @author Max Pumperla
 */
@DisplayName("Keras Zero Padding Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
class KerasZeroPaddingTest extends BaseDL4JTest {

    private final Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private final Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    static Collection<Object[]> dimensionalities() {
        return Arrays.asList(new Object[][]{{1}, {2}, {3}});
    }

    // -------------------------------------------------------------------------
    // ZeroPadding1D
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Zero Padding 1D Layer")
    void testZeroPadding1DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildZeroPadding1DLayer(conf, kerasVersion);
    }

    private void buildZeroPadding1DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        String layerName = "zero_padding_1D_layer";
        int zeroPadding = 2;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_1D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), zeroPadding);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPadding1DLayer layer = new KerasZeroPadding1D(layerConfig).getZeroPadding1DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(zeroPadding, layer.getPadding()[0]);
    }

    // -------------------------------------------------------------------------
    // ZeroPadding2D — multi-dim variant
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Zero Padding 2D Layer (multi-dim)")
    void testZeroPadding2DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildZeroPadding2DLayer(conf, kerasVersion);
    }

    private void buildZeroPadding2DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "zero_padding_2D_layer";
        final int[] zeroPadding = new int[]{2, 3};

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        ArrayList<Integer> padding = new ArrayList<>();
        for (int i : zeroPadding) padding.add(i);
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), padding);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPaddingLayer layer = new KerasZeroPadding2D(layerConfig).getZeroPadding2DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(zeroPadding[0], layer.getPadding()[0]);
        assertEquals(zeroPadding[0], layer.getPadding()[1]);
        assertEquals(zeroPadding[1], layer.getPadding()[2]);
        assertEquals(zeroPadding[1], layer.getPadding()[3]);
    }

    // -------------------------------------------------------------------------
    // ZeroPadding2D — single-dim variant
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Zero Padding 2D Layer (single-dim)")
    void testZeroPaddingSingleDim2DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildZeroPaddingSingleDim2DLayer(conf, kerasVersion);
    }

    private void buildZeroPaddingSingleDim2DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "zero_padding_2D_layer";
        final int zeroPaddingScalar = 2;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), zeroPaddingScalar);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPaddingLayer layer = new KerasZeroPadding2D(layerConfig).getZeroPadding2DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(zeroPaddingScalar, layer.getPadding()[0]);
        assertEquals(zeroPaddingScalar, layer.getPadding()[1]);
    }

    // -------------------------------------------------------------------------
    // ZeroPadding3D — multi-dim variant
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Zero Padding 3D Layer (multi-dim)")
    void testZeroPadding3DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildZeroPadding3DLayer(conf, kerasVersion);
    }

    private void buildZeroPadding3DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "zero_padding_3D_layer";
        final int[] zeroPadding = new int[]{2, 3, 4};

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_3D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        List<Integer> padding = new ArrayList<>();
        for (int i : zeroPadding) padding.add(i);
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), padding);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPadding3DLayer layer = new KerasZeroPadding3D(layerConfig).getZeroPadding3DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(zeroPadding[0], layer.getPadding()[0]);
        assertEquals(zeroPadding[0], layer.getPadding()[1]);
        assertEquals(zeroPadding[1], layer.getPadding()[2]);
        assertEquals(zeroPadding[1], layer.getPadding()[3]);
        assertEquals(zeroPadding[2], layer.getPadding()[4]);
        assertEquals(zeroPadding[2], layer.getPadding()[5]);
    }

    // -------------------------------------------------------------------------
    // ZeroPadding3D — single-dim variant
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Zero Padding 3D Layer (single-dim)")
    void testZeroPaddingSingleDim3DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildZeroPaddingSingleDim3DLayer(conf, kerasVersion);
    }

    private void buildZeroPaddingSingleDim3DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "zero_padding_3D_layer";
        final int zeroPaddingScalar = 2;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_ZERO_PADDING_3D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        config.put(conf.getLAYER_FIELD_ZERO_PADDING(), zeroPaddingScalar);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ZeroPadding3DLayer layer = new KerasZeroPadding3D(layerConfig).getZeroPadding3DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(zeroPaddingScalar, layer.getPadding()[0]);
        assertEquals(zeroPaddingScalar, layer.getPadding()[1]);
    }

    // -------------------------------------------------------------------------
    // Parameter source
    // -------------------------------------------------------------------------

    static Collection<Object[]> kerasVersions() {
        return Arrays.asList(new Object[][]{{1}, {2}});
    }
}
