/*******************************************************************************
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

package org.deeplearning4j.nn.modelimport.keras.layers.convolution;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolution2D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasDepthwiseConvolution2D;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.base.Preconditions;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasDepthwiseConvolution2DTest {

    private final String ACTIVATION_KERAS = "linear";
    private final String ACTIVATION_DL4J = "identity";
    private final String LAYER_NAME = "test_layer";
    private final String INIT_KERAS = "depthwise_conv_2d";
    private final WeightInit INIT_DL4J = WeightInit.XAVIER;
    private final double L1_REGULARIZATION = 0.01;
    private final double L2_REGULARIZATION = 0.02;
    private final double DROPOUT_KERAS = 0.3;
    private final double DROPOUT_DL4J = 1 - DROPOUT_KERAS;
    private final int[] KERNEL_SIZE = new int[]{1, 2};
    private final int[] DILATION = new int[]{2, 2};
    private final int[] STRIDE = new int[]{3, 4};
    private final int DEPTH_MULTIPLIER = 4;
    private final int N_IN = 3;
    private final String BORDER_MODE_VALID = "valid";
    private final int[] VALID_PADDING = new int[]{0, 0};

    private Integer keras2 = 2;
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();


    @Test
    public void testDepthwiseConvolution2DLayer() throws Exception {
        buildDepthwiseConvolution2DLayer(conf2, keras2, false);
        buildDepthwiseConvolution2DLayer(conf2, keras2, true);
    }


    private void buildDepthwiseConvolution2DLayer(KerasLayerConfiguration conf, Integer kerasVersion, boolean withDilation)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_DEPTHWISE_CONVOLUTION_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS);
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);
            config.put(conf.getLAYER_FIELD_DEPTH_WISE_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_INIT(), init);
            config.put(conf.getLAYER_FIELD_DEPTH_WISE_INIT(), init);
        }
        Map<String, Object> W_reg = new HashMap<>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        config.put(conf.getLAYER_FIELD_DEPTH_WISE_REGULARIZER(), W_reg);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        config.put(conf.getLAYER_FIELD_DEPTH_MULTIPLIER(), DEPTH_MULTIPLIER);

        ArrayList kernel = new ArrayList<Integer>() {{
            for (int i : KERNEL_SIZE) add(i);
        }};
        config.put(conf.getLAYER_FIELD_KERNEL_SIZE(), kernel);

        if (withDilation) {
            ArrayList dilation = new ArrayList<Integer>() {{
                for (int i : DILATION) add(i);
            }};
            config.put(conf.getLAYER_FIELD_DILATION_RATE(), dilation);
        }
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put(conf.getLAYER_FIELD_CONVOLUTION_STRIDES(), subsampleList);
        config.put(conf.getLAYER_FIELD_BORDER_MODE(), BORDER_MODE_VALID);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);
        config.put(conf.getLAYER_FIELD_NB_FILTER(), N_IN);

        KerasConvolution2D previousLayer = new KerasConvolution2D(layerConfig);
        Map<String, KerasLayer> previousLayers = new HashMap<>();
        previousLayers.put("conv", previousLayer);
        List<String> layerNames = Collections.singletonList("conv");

        KerasDepthwiseConvolution2D kerasLayer =  new KerasDepthwiseConvolution2D(
                layerConfig, previousLayers, layerNames, false);
        Preconditions.checkState(kerasLayer.getInboundLayerNames().get(0).equalsIgnoreCase("conv"), "Expected inbound name to be \"conv\" - was \"%s\"", kerasLayer.getInboundLayerNames().get(0));

        DepthwiseConvolution2D layer = kerasLayer.getDepthwiseConvolution2DLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(DEPTH_MULTIPLIER, layer.getDepthMultiplier());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(new Dropout(DROPOUT_DL4J), layer.getIDropout());
        assertArrayEquals(KERNEL_SIZE, layer.getKernelSize());
        assertArrayEquals(STRIDE, layer.getStride());
        assertEquals(N_IN * DEPTH_MULTIPLIER, layer.getNOut());
        assertEquals(ConvolutionMode.Truncate, layer.getConvolutionMode());
        assertArrayEquals(VALID_PADDING, layer.getPadding());
        if (withDilation) {
            assertEquals(DILATION[0], layer.getDilation()[0]);
            assertEquals(DILATION[1], layer.getDilation()[1]);
        }
    }
}
