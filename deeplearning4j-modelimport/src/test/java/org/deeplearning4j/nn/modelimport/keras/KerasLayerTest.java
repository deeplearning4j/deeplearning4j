/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.advanced.activations.KerasLeakyReLU;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasAtrousConvolution1D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolution1D;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolution2D;
import org.deeplearning4j.nn.modelimport.keras.layers.core.KerasActivation;
import org.deeplearning4j.nn.modelimport.keras.layers.core.KerasDense;
import org.deeplearning4j.nn.modelimport.keras.layers.core.KerasDropout;
import org.deeplearning4j.nn.modelimport.keras.layers.core.KerasReshape;
import org.deeplearning4j.nn.modelimport.keras.layers.embeddings.KerasEmbedding;
import org.deeplearning4j.nn.modelimport.keras.layers.normalization.KerasBatchNormalization;
import org.deeplearning4j.nn.modelimport.keras.layers.pooling.KerasPooling1D;
import org.deeplearning4j.nn.modelimport.keras.layers.pooling.KerasPooling2D;
import org.deeplearning4j.nn.modelimport.keras.layers.recurrent.KerasLstm;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Unit tests for end-to-end Keras 1.x layer configuration import.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasLayerTest {
    private final String ACTIVATION_KERAS = "linear";
    private final String ACTIVATION_DL4J = "identity";
    private final String LAYER_NAME = "test_layer";
    private final String INIT_KERAS = "glorot_normal";
    private final WeightInit INIT_DL4J = WeightInit.XAVIER;
    private final double L1_REGULARIZATION = 0.01;
    private final double L2_REGULARIZATION = 0.02;
    private final double DROPOUT_KERAS = 0.3;
    private final double DROPOUT_DL4J = 1 - DROPOUT_KERAS;
    private final int[] KERNEL_SIZE = new int[]{1, 2};
    private final int[] DILATION = new int[]{2, 2};
    private final int[] INPUT_SHAPE = new int[]{100, 20};
    private final int[] STRIDE = new int[]{3, 4};
    private final PoolingType POOLING_TYPE = PoolingType.MAX;
    private final int N_OUT = 13;
    private final String BORDER_MODE_VALID = "valid";
    private final int[] VALID_PADDING = new int[]{0, 0};

    private Integer keras1 = 1;
    private Integer keras2 = 2;
    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    public KerasLayerTest() throws UnsupportedKerasConfigurationException {
    }

    @Test
    public void testEmbeddingLayer() throws Exception {
        buildEmbeddingLayer(conf1, keras1);
        buildEmbeddingLayer(conf2, keras2);
    }

    @Test
    public void testActivationLayer() throws Exception {
        buildActivationLayer(conf1, keras1);
        buildActivationLayer(conf2, keras2);
    }

    @Test
    public void testAtrousConvolution1DLayer() throws Exception {
        buildAtrousConvolution1DLayer(conf1, keras1);
    }

    @Test
    public void testAtrousConvolution2DLayer() throws Exception {
        buildAtrousConvolution2DLayer(conf1, keras1);
    }


    @Test
    public void testConvolution2DLayer() throws Exception {
        buildConvolution2DLayer(conf1, keras1, false);
        buildConvolution2DLayer(conf2, keras2, false);
        buildConvolution2DLayer(conf2, keras2, true);
    }

    @Test
    public void testConvolution1DLayer() throws Exception {
        buildConvolution1DLayer(conf1, keras1, false);
        buildConvolution1DLayer(conf2, keras2, false);
        buildConvolution1DLayer(conf2, keras2, true);
    }

    @Test
    public void testSubsampling2DLayer() throws Exception {
        buildSubsampling2DLayer(conf1, keras1);
        buildSubsampling2DLayer(conf2, keras2);
    }

    @Test
    public void testSubsampling1DLayer() throws Exception {
        buildSubsampling1DLayer(conf1, keras1);
        buildSubsampling1DLayer(conf2, keras2);
    }

    @Test
    public void testDenseLayer() throws Exception {
        buildDenseLayer(conf1, keras1);
        buildDenseLayer(conf2, keras2);
    }

    @Test
    public void testGravesLstmLayer() throws Exception {
        buildGravesLstmLayer(conf1, keras1);
        buildGravesLstmLayer(conf2, keras2);
    }

    @Test
    public void testDropoutLayer() throws Exception {
        buildDropoutLayer(conf1, keras1);
        buildDropoutLayer(conf2, keras2);
    }

    @Test
    public void testBatchnormLayer() throws Exception {
        buildBatchNormalizationLayer(conf1, keras1);
        buildBatchNormalizationLayer(conf2, keras2);
    }

    @Test
    public void testLeakyReLULayer() throws Exception {
        buildLeakyReLULayer(conf1, keras1);
        buildLeakyReLULayer(conf2, keras2);
    }

    @Test
    public void testReshapeLayer() throws Exception {
        buildLReshapeLayer(conf1, keras1);
        buildLReshapeLayer(conf2, keras2);
    }

    public void buildConvolution1DLayer(KerasLayerConfiguration conf, Integer kerasVersion, boolean withDilation)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CONVOLUTION_1D());
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS);
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<String, Object>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        if (withDilation){
            config.put(conf.getLAYER_FIELD_DILATION_RATE(), DILATION[0]);
        }
        Map<String, Object> W_reg = new HashMap<String, Object>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        config.put(conf.getLAYER_FIELD_W_REGULARIZER(), W_reg);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        config.put(conf.getLAYER_FIELD_FILTER_LENGTH(), KERNEL_SIZE[0]);
        config.put(conf.getLAYER_FIELD_SUBSAMPLE_LENGTH(), STRIDE[0]);
        config.put(conf.getLAYER_FIELD_NB_FILTER(), N_OUT);
        config.put(conf.getLAYER_FIELD_BORDER_MODE(), BORDER_MODE_VALID);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);

        Convolution1DLayer layer = new KerasConvolution1D(layerConfig).getConvolution1DLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertEquals(KERNEL_SIZE[0], layer.getKernelSize()[0]);
        assertEquals(STRIDE[0], layer.getStride()[0]);
        assertEquals(N_OUT, layer.getNOut());
        assertEquals(ConvolutionMode.Truncate, layer.getConvolutionMode());
        assertEquals(VALID_PADDING[0], layer.getPadding()[0]);
        if (withDilation) {
            assertEquals(DILATION[0], layer.getDilation()[0]);
        }
    }

    public void buildAtrousConvolution1DLayer(KerasLayerConfiguration conf, Integer kerasVersion)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CONVOLUTION_1D());
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS);
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<String, Object>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        config.put(conf.getLAYER_FIELD_DILATION_RATE(), DILATION[0]);
        Map<String, Object> W_reg = new HashMap<String, Object>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        config.put(conf.getLAYER_FIELD_W_REGULARIZER(), W_reg);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        config.put(conf.getLAYER_FIELD_FILTER_LENGTH(), KERNEL_SIZE[0]);
        config.put(conf.getLAYER_FIELD_SUBSAMPLE_LENGTH(), STRIDE[0]);
        config.put(conf.getLAYER_FIELD_NB_FILTER(), N_OUT);
        config.put(conf.getLAYER_FIELD_BORDER_MODE(), BORDER_MODE_VALID);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);

        Convolution1DLayer layer = new KerasAtrousConvolution1D(layerConfig).getAtrousConvolution1D();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertEquals(KERNEL_SIZE[0], layer.getKernelSize()[0]);
        assertEquals(STRIDE[0], layer.getStride()[0]);
        assertEquals(N_OUT, layer.getNOut());
        assertEquals(ConvolutionMode.Truncate, layer.getConvolutionMode());
        assertEquals(VALID_PADDING[0], layer.getPadding()[0]);
        assertEquals(DILATION[0], layer.getDilation()[0]);
    }

    public void buildEmbeddingLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_EMBEDDING());
        Map<String, Object> config = new HashMap<String, Object>();
        Integer inputDim = 10;
        Integer outputDim = 10;
        config.put(conf.getLAYER_FIELD_INPUT_DIM(), inputDim);
        config.put(conf.getLAYER_FIELD_OUTPUT_DIM(), outputDim);

        ArrayList inputShape = new ArrayList<Integer>() {{
            for (int i : INPUT_SHAPE) add(i);
        }};
        config.put(conf.getLAYER_FIELD_BATCH_INPUT_SHAPE(), inputShape);
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_EMBEDDING_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<String, Object>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_EMBEDDING_INIT(), init);
        }
        KerasEmbedding kerasEmbedding = new KerasEmbedding(layerConfig, false);
        assertEquals(kerasEmbedding.getNumParams(), 1);

        EmbeddingLayer layer = kerasEmbedding.getEmbeddingLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());

    }

    public void buildActivationLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_FIELD_ACTIVATION());
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ActivationLayer layer = new KerasActivation(layerConfig).getActivationLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
    }

    public void buildLeakyReLULayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_LEAKY_RELU());
        Map<String, Object> config = new HashMap<String, Object>();
        String LAYER_FIELD_LEAKY_RELU_ALPHA = "alpha";
        config.put(LAYER_FIELD_LEAKY_RELU_ALPHA, 0.3); // set leaky ReLU alpha
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        ActivationLayer layer = new KerasLeakyReLU(layerConfig).getActivationLayer();
        assertEquals("leakyrelu(a=0.3)", layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
    }

    public void buildLReshapeLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_RESHAPE());
        Map<String, Object> config = new HashMap<String, Object>();
        int[] targetShape = new int[]{10, 5};
        List<Integer> targetShapeList = new ArrayList<>();
        targetShapeList.add(targetShape[0]);
        targetShapeList.add(targetShape[1]);
        String LAYER_FIELD_TARGET_SHAPE = "target_shape";
        config.put(LAYER_FIELD_TARGET_SHAPE, targetShapeList);
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        InputType inputType = InputType.InputTypeFeedForward.feedForward(20);
        ReshapePreprocessor preProcessor =
                (ReshapePreprocessor) new KerasReshape(layerConfig).getInputPreprocessor(inputType);
        assertEquals(preProcessor.getTargetShape()[0], targetShape[0]);
        assertEquals(preProcessor.getTargetShape()[1], targetShape[1]);
    }

    public void buildDropoutLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_DROPOUT());
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        DropoutLayer layer = new KerasDropout(layerConfig).getDropoutLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
    }

    public void buildDenseLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_DENSE());
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<String, Object>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        Map<String, Object> W_reg = new HashMap<String, Object>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        config.put(conf.getLAYER_FIELD_W_REGULARIZER(), W_reg);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        config.put(conf.getLAYER_FIELD_OUTPUT_DIM(), N_OUT);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        DenseLayer layer = new KerasDense(layerConfig, false).getDenseLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertEquals(N_OUT, layer.getNOut());
    }

    public void buildConvolution2DLayer(KerasLayerConfiguration conf, Integer kerasVersion, boolean withDilation)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CONVOLUTION_2D());
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<String, Object>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        Map<String, Object> W_reg = new HashMap<String, Object>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        config.put(conf.getLAYER_FIELD_W_REGULARIZER(), W_reg);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_NB_ROW(), KERNEL_SIZE[0]);
            config.put(conf.getLAYER_FIELD_NB_COL(), KERNEL_SIZE[1]);
        } else {
            ArrayList kernel = new ArrayList<Integer>() {{
                for (int i : KERNEL_SIZE) add(i);
            }};
            config.put(conf.getLAYER_FIELD_KERNEL_SIZE(), kernel);
        }
        if (withDilation){
            ArrayList dilation = new ArrayList<Integer>() {{
                for (int i : DILATION) add(i);
            }};
            config.put(conf.getLAYER_FIELD_DILATION_RATE(), dilation);
        }
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put(conf.getLAYER_FIELD_CONVOLUTION_STRIDES(), subsampleList);
        config.put(conf.getLAYER_FIELD_NB_FILTER(), N_OUT);
        config.put(conf.getLAYER_FIELD_BORDER_MODE(), BORDER_MODE_VALID);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);


        ConvolutionLayer layer = new KerasConvolution2D(layerConfig).getConvolution2DLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertArrayEquals(KERNEL_SIZE, layer.getKernelSize());
        assertArrayEquals(STRIDE, layer.getStride());
        assertEquals(N_OUT, layer.getNOut());
        assertEquals(ConvolutionMode.Truncate, layer.getConvolutionMode());
        assertArrayEquals(VALID_PADDING, layer.getPadding());
        if (withDilation) {
            assertEquals(DILATION[0], layer.getDilation()[0]);
            assertEquals(DILATION[1], layer.getDilation()[1]);
        }
    }

    public void buildAtrousConvolution2DLayer(KerasLayerConfiguration conf, Integer kerasVersion)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_CONVOLUTION_2D());
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<String, Object>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        Map<String, Object> W_reg = new HashMap<String, Object>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        config.put(conf.getLAYER_FIELD_W_REGULARIZER(), W_reg);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_NB_ROW(), KERNEL_SIZE[0]);
            config.put(conf.getLAYER_FIELD_NB_COL(), KERNEL_SIZE[1]);
        } else {
            ArrayList kernel = new ArrayList<Integer>() {{
                for (int i : KERNEL_SIZE) add(i);
            }};
            config.put(conf.getLAYER_FIELD_KERNEL_SIZE(), kernel);
        }
        ArrayList dilation = new ArrayList<Integer>() {{
            for (int i : DILATION) add(i);
        }};
        config.put(conf.getLAYER_FIELD_DILATION_RATE(), dilation);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put(conf.getLAYER_FIELD_CONVOLUTION_STRIDES(), subsampleList);
        config.put(conf.getLAYER_FIELD_NB_FILTER(), N_OUT);
        config.put(conf.getLAYER_FIELD_BORDER_MODE(), BORDER_MODE_VALID);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);


        ConvolutionLayer layer = new KerasConvolution2D(layerConfig).getConvolution2DLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertArrayEquals(KERNEL_SIZE, layer.getKernelSize());
        assertArrayEquals(STRIDE, layer.getStride());
        assertEquals(N_OUT, layer.getNOut());
        assertEquals(ConvolutionMode.Truncate, layer.getConvolutionMode());
        assertArrayEquals(VALID_PADDING, layer.getPadding());
        assertEquals(DILATION[0], layer.getDilation()[0]);
        assertEquals(DILATION[1], layer.getDilation()[1]);
    }

    public void buildSubsampling2DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_MAX_POOLING_2D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        List<Integer> kernelSizeList = new ArrayList<>();
        kernelSizeList.add(KERNEL_SIZE[0]);
        kernelSizeList.add(KERNEL_SIZE[1]);
        config.put(conf.getLAYER_FIELD_POOL_SIZE(), kernelSizeList);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put(conf.getLAYER_FIELD_POOL_STRIDES(), subsampleList);
        config.put(conf.getLAYER_FIELD_BORDER_MODE(), BORDER_MODE_VALID);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        SubsamplingLayer layer = new KerasPooling2D(layerConfig).getSubsampling2DLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertArrayEquals(KERNEL_SIZE, layer.getKernelSize());
        assertArrayEquals(STRIDE, layer.getStride());
        assertEquals(POOLING_TYPE, layer.getPoolingType());
        assertEquals(ConvolutionMode.Truncate, layer.getConvolutionMode());
        assertArrayEquals(VALID_PADDING, layer.getPadding());
    }

    public void buildSubsampling1DLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_MAX_POOLING_1D());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put(conf.getLAYER_FIELD_POOL_1D_SIZE(), KERNEL_SIZE[0]);
        config.put(conf.getLAYER_FIELD_POOL_1D_STRIDES(), STRIDE[0]);
        config.put(conf.getLAYER_FIELD_BORDER_MODE(), BORDER_MODE_VALID);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        Subsampling1DLayer layer = new KerasPooling1D(layerConfig).getSubsampling1DLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(KERNEL_SIZE[0], layer.getKernelSize()[0]);
        assertEquals(STRIDE[0], layer.getStride()[0]);
        assertEquals(POOLING_TYPE, layer.getPoolingType());
        assertEquals(ConvolutionMode.Truncate, layer.getConvolutionMode());
        assertEquals(VALID_PADDING[0], layer.getPadding()[0]);
    }

    public void buildGravesLstmLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        String innerActivation = "hard_sigmoid";
        double lstmForgetBiasDouble = 1.0;
        String lstmForgetBiasString = "one";
        boolean lstmUnroll = true;

        KerasLstm lstm = new KerasLstm(kerasVersion);

        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_LSTM());
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(conf.getLAYER_FIELD_INNER_ACTIVATION(), innerActivation); // keras linear -> dl4j identity
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INNER_INIT(), INIT_KERAS);
            config.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);

        } else {
            Map<String, Object> init = new HashMap<String, Object>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_INNER_INIT(), init);
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        Map<String, Object> W_reg = new HashMap<String, Object>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        config.put(conf.getLAYER_FIELD_W_REGULARIZER(), W_reg);
        config.put(conf.getLAYER_FIELD_DROPOUT_W(), DROPOUT_KERAS);
        config.put(conf.getLAYER_FIELD_DROPOUT_U(), 0.0);
        config.put(conf.getLAYER_FIELD_FORGET_BIAS_INIT(), lstmForgetBiasString);
        config.put(conf.getLAYER_FIELD_OUTPUT_DIM(), N_OUT);
        config.put(lstm.getLAYER_FIELD_UNROLL(), lstmUnroll);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        GravesLSTM layer = new KerasLstm(layerConfig).getGravesLSTMLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertEquals(lstmForgetBiasDouble, layer.getForgetGateBiasInit(), 0.0);
        assertEquals(N_OUT, layer.getNOut());
    }

    public void buildBatchNormalizationLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        double epsilon = 1E-5;
        double momentum = 0.99;

        KerasBatchNormalization batchNormalization = new KerasBatchNormalization(kerasVersion);

        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_BATCHNORMALIZATION());
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        config.put(batchNormalization.getLAYER_FIELD_EPSILON(), epsilon);
        config.put(batchNormalization.getLAYER_FIELD_MOMENTUM(), momentum);
        config.put(batchNormalization.getLAYER_FIELD_GAMMA_REGULARIZER(), null);
        config.put(batchNormalization.getLAYER_FIELD_BETA_REGULARIZER(), null);
        config.put(batchNormalization.getLAYER_FIELD_MODE(), 0);
        config.put(batchNormalization.getLAYER_FIELD_AXIS(), 3);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        BatchNormalization layer = new KerasBatchNormalization(layerConfig).getBatchNormalizationLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(epsilon, layer.getEps(), 0.0);
        assertEquals(momentum, layer.getMomentum(), 0.0);
    }
}
