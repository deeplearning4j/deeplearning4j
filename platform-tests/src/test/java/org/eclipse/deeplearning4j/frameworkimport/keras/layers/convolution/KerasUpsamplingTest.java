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

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.Upsampling1D;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.layers.Upsampling3D;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasUpsampling1D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasUpsampling2D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasUpsampling3D;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Parameterized Keras Upsampling layer tests covering 1D, 2D, and 3D variants.
 * Collapses KerasUpsampling1DTest, KerasUpsampling2DTest, and KerasUpsampling3DTest
 * into a single class. The 3D variant additionally iterates over all dim orderings.
 *
 * @author Max Pumperla
 */
@DisplayName("Keras Upsampling Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
class KerasUpsamplingTest extends BaseDL4JTest {

    private final Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private final Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    // -------------------------------------------------------------------------
    // 1D
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "Upsampling1D kerasVersion={0}")
    @ValueSource(ints = {1, 2})
    @DisplayName("Test Upsampling 1D Layer")
    void testUpsampling1DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildUpsampling1DLayer(conf, kerasVersion);
    }

    private void buildUpsampling1DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "upsampling_1D_layer";
        final int size = 4;

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_UPSAMPLING_1D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_UPSAMPLING_1D_SIZE(), size);
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Upsampling1D layer = new KerasUpsampling1D(layerConfig).getUpsampling1DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(size, layer.getSize()[0]);
    }

    // -------------------------------------------------------------------------
    // 2D
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "Upsampling2D kerasVersion={0}")
    @ValueSource(ints = {1, 2})
    @DisplayName("Test Upsampling 2D Layer")
    void testUpsampling2DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        buildUpsampling2DLayer(conf, kerasVersion);
    }

    private void buildUpsampling2DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        final String layerName = "upsampling_2D_layer";
        final int[] size = new int[]{2, 2};

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_UPSAMPLING_2D());
        Map<String, Object> config = new HashMap<>();
        List<Integer> sizeList = new ArrayList<>();
        sizeList.add(size[0]);
        sizeList.add(size[1]);
        config.put(conf.getLAYER_FIELD_UPSAMPLING_2D_SIZE(), sizeList);
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Upsampling2D layer = new KerasUpsampling2D(layerConfig).getUpsampling2DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(size[0], layer.getSize()[0]);
        assertEquals(size[1], layer.getSize()[1]);
    }

    // -------------------------------------------------------------------------
    // 3D — iterates over all DimOrder values, mirroring the original loop
    // -------------------------------------------------------------------------

    @ParameterizedTest(name = "Upsampling3D kerasVersion={0}")
    @ValueSource(ints = {1, 2})
    @DisplayName("Test Upsampling 3D Layer")
    void testUpsampling3DLayer(int kerasVersion) throws Exception {
        KerasLayerConfiguration conf = kerasVersion == 1 ? conf1 : conf2;
        for (KerasLayer.DimOrder dimOrder : KerasLayer.DimOrder.values()) {
            String ordering = dimOrder != KerasLayer.DimOrder.THEANO ? "channels_last" : "channels_first";
            buildUpsampling3DLayer(conf, kerasVersion, ordering);
        }
    }

    private void buildUpsampling3DLayer(KerasLayerConfiguration conf, Integer kerasVersion, String ordering) throws Exception {
        final String layerName = "upsampling_3D_layer";
        final int[] size = new int[]{2, 2, 2};

        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_UPSAMPLING_3D());
        Map<String, Object> config = new HashMap<>();
        List<Integer> sizeList = new ArrayList<>();
        sizeList.add(size[0]);
        sizeList.add(size[1]);
        sizeList.add(size[2]);
        config.put(conf.getLAYER_FIELD_UPSAMPLING_3D_SIZE(), sizeList);
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        config.put(conf.getLAYER_FIELD_DIM_ORDERING(), ordering);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Upsampling3D layer = new KerasUpsampling3D(layerConfig).getUpsampling3DLayer();
        assertEquals(layerName, layer.getLayerName());
        assertEquals(size[0], layer.getSize()[0]);
        assertEquals(size[1], layer.getSize()[1]);
        assertEquals(size[2], layer.getSize()[2]);
        if (ordering.equals("channels_last")) {
            assertEquals(Convolution3D.DataFormat.NDHWC, layer.getDataFormat());
        } else if (ordering.equals("channels_first")) {
            assertEquals(Convolution3D.DataFormat.NCDHW, layer.getDataFormat());
        }
    }
}
