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

import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping3D;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasCropping1D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasCropping2D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasCropping3D;
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
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Parameterized test covering Cropping1D, Cropping2D, and Cropping3D Keras layer import.
 *
 * @author Max Pumperla
 */
@DisplayName("Keras Cropping Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
class KerasCroppingTest extends BaseDL4JTest {

    private final Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private final Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    // -------------------------------------------------------------------------
    // Cropping1D
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Cropping 1D Layer")
    void testCropping1DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildCroppingSingleDim1DLayer(conf, kerasVersion);
    }

    private void buildCroppingSingleDim1DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "cropping_1D_layer";
        final int cropping = 2;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CROPPING_1D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        config.put(conf.getLAYER_FIELD_CROPPING(), cropping);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Cropping1D layer = new KerasCropping1D(layerConfig).getCropping1DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(cropping, layer.getCropping()[0]);
        assertEquals(cropping, layer.getCropping()[1]);
    }

    // -------------------------------------------------------------------------
    // Cropping2D — multi-dim variant
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Cropping 2D Layer (multi-dim)")
    void testCropping2DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildCropping2DLayer(conf, kerasVersion);
    }

    private void buildCropping2DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "cropping_2D_layer";
        final int[] cropping = new int[]{2, 3};

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CROPPING_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        ArrayList<Integer> paddingList = new ArrayList<>();
        for (int i : cropping) paddingList.add(i);
        config.put(conf.getLAYER_FIELD_CROPPING(), paddingList);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Cropping2D layer = new KerasCropping2D(layerConfig).getCropping2DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(cropping[0], layer.getCropping()[0]);
        assertEquals(cropping[0], layer.getCropping()[1]);
        assertEquals(cropping[1], layer.getCropping()[2]);
        assertEquals(cropping[1], layer.getCropping()[3]);
    }

    // -------------------------------------------------------------------------
    // Cropping2D — single-dim variant
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Cropping 2D Layer (single-dim)")
    void testCroppingSingleDim2DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildCroppingSingleDim2DLayer(conf, kerasVersion);
    }

    private void buildCroppingSingleDim2DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "cropping_2D_layer";
        final int croppingScalar = 2;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CROPPING_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        config.put(conf.getLAYER_FIELD_CROPPING(), croppingScalar);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Cropping2D layer = new KerasCropping2D(layerConfig).getCropping2DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(croppingScalar, layer.getCropping()[0]);
        assertEquals(croppingScalar, layer.getCropping()[1]);
    }

    // -------------------------------------------------------------------------
    // Cropping3D — multi-dim variant
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Cropping 3D Layer (multi-dim)")
    void testCropping3DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildCropping3DLayer(conf, kerasVersion);
    }

    private void buildCropping3DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "cropping_3D_layer";
        final int[] cropping = new int[]{2, 3, 5};

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CROPPING_3D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        ArrayList<Integer> paddingList = new ArrayList<>();
        for (int i : cropping) paddingList.add(i);
        config.put(conf.getLAYER_FIELD_CROPPING(), paddingList);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Cropping3D layer = new KerasCropping3D(layerConfig).getCropping3DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(cropping[0], layer.getCropping()[0]);
        assertEquals(cropping[0], layer.getCropping()[1]);
        assertEquals(cropping[1], layer.getCropping()[2]);
        assertEquals(cropping[1], layer.getCropping()[3]);
        assertEquals(cropping[2], layer.getCropping()[4]);
        assertEquals(cropping[2], layer.getCropping()[5]);
    }

    // -------------------------------------------------------------------------
    // Cropping3D — single-dim variant
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "kerasVersion={0}")
    @MethodSource("kerasVersions")
    @DisplayName("Test Cropping 3D Layer (single-dim)")
    void testCroppingSingleDim3DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildCroppingSingleDim3DLayer(conf, kerasVersion);
    }

    private void buildCroppingSingleDim3DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "cropping_3D_layer";
        final int croppingScalar = 2;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CROPPING_3D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        config.put(conf.getLAYER_FIELD_CROPPING(), croppingScalar);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Cropping3D layer = new KerasCropping3D(layerConfig).getCropping3DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(croppingScalar, layer.getCropping()[0]);
        assertEquals(croppingScalar, layer.getCropping()[1]);
    }

    // -------------------------------------------------------------------------
    // Parameter source
    // -------------------------------------------------------------------------

    static Collection<Object[]> kerasVersions() {
        return Arrays.asList(new Object[][]{{1}, {2}});
    }
}
