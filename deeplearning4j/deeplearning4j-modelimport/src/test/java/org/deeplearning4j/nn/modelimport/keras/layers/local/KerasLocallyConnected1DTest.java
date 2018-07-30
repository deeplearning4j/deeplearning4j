/*
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.deeplearning4j.nn.modelimport.keras.layers.local;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LocallyConnected1D;
import org.deeplearning4j.nn.conf.layers.LocallyConnected2D;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasLocallyConnected1DTest {

    private final String ACTIVATION_KERAS = "linear";
    private final String ACTIVATION_DL4J = "identity";
    private final String LAYER_NAME = "test_layer";
    private final String INIT_KERAS = "glorot_normal";
    private final WeightInit INIT_DL4J = WeightInit.XAVIER;
    private final double L1_REGULARIZATION = 0.01;
    private final double L2_REGULARIZATION = 0.02;
    private final double DROPOUT_KERAS = 0.3;
    private final double DROPOUT_DL4J = 1 - DROPOUT_KERAS;
    private final int KERNEL_SIZE = 2;
    private final int STRIDE = 3;
    private final int N_OUT = 13;
    private final String BORDER_MODE_VALID = "valid";
    private final int VALID_PADDING = 0;

    private Integer keras1 = 1;
    private Integer keras2 = 2;
    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();


    @Test
    public void testLocallyConnected2DLayer() throws Exception {
        buildLocallyConnected2DLayer(conf1, keras1);
        buildLocallyConnected2DLayer(conf2, keras2);
    }


    private void buildLocallyConnected2DLayer(KerasLayerConfiguration conf, Integer kerasVersion)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_LOCALLY_CONNECTED_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS);
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        Map<String, Object> W_reg = new HashMap<>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        config.put(conf.getLAYER_FIELD_W_REGULARIZER(), W_reg);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        if (kerasVersion == 2) {
            ArrayList kernel = new ArrayList<Integer>() {{
                add(KERNEL_SIZE);
            }};
            config.put(conf.getLAYER_FIELD_FILTER_LENGTH(), kernel);
        } else {
            config.put(conf.getLAYER_FIELD_FILTER_LENGTH(), KERNEL_SIZE);
        }

        if (kerasVersion == 2) {
            ArrayList stride = new ArrayList<Integer>() {{
                add(STRIDE);
            }};
            config.put(conf.getLAYER_FIELD_SUBSAMPLE_LENGTH(), stride);
        } else {
            config.put(conf.getLAYER_FIELD_SUBSAMPLE_LENGTH(), STRIDE);
        }

        config.put(conf.getLAYER_FIELD_NB_FILTER(), N_OUT);
        config.put(conf.getLAYER_FIELD_BORDER_MODE(), BORDER_MODE_VALID);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);


        KerasLocallyConnected1D kerasLocal = new KerasLocallyConnected1D(layerConfig);

        // once get output type is triggered, inputshape, output shape and input depth are updated
        kerasLocal.getOutputType(InputType.recurrent(3,  4));

        LocallyConnected1D layer = kerasLocal.getLocallyConnected1DLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivation().toString().toLowerCase());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(new Dropout(DROPOUT_DL4J), layer.getIDropout());
        assertEquals(KERNEL_SIZE, layer.getKernel());
        assertEquals(STRIDE, layer.getStride());
        assertEquals(N_OUT, layer.getNOut());
        assertEquals(ConvolutionMode.Truncate, layer.getCm());
        assertEquals(VALID_PADDING, layer.getPadding());

        assertEquals(layer.getInputSize(), 4);
        assertEquals(layer.getNIn(), 3);
    }
}

