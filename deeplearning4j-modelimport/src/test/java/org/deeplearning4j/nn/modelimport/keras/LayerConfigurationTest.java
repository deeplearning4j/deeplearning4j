package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.*;

import static org.deeplearning4j.nn.modelimport.keras.LayerConfiguration.*;
import static org.junit.Assert.*;

/**
 * Unit tests for end-to-end Keras layer configuration import.
 *
 * @author davekale
 */
public class LayerConfigurationTest {
    public static final String ACTIVATION_KERAS = "linear";
    public static final String ACTIVATION_DL4J = "identity";
    public static final String LAYER_NAME = "test_layer";
    public static final String INIT_KERAS = "glorot_normal";
    public static final WeightInit INIT_DL4J = WeightInit.XAVIER;
    public static final double L1_REGULARIZATION = 0.01;
    public static final double L2_REGULARIZATION = 0.02;
    public static final double DROPOUT_KERAS = 0.3;
    public static final double DROPOUT_DL4J = 1-DROPOUT_KERAS;
    public static final int[] KERNEL_SIZE = new int[]{1, 2};
    public static final int[] STRIDE = new int[]{3, 4};
    public static final SubsamplingLayer.PoolingType POOLING_TYPE = SubsamplingLayer.PoolingType.MAX;
    public static final double LSTM_FORGET_BIAS_DOUBLE = 1.0;
    public static final String LSTM_FORGET_BIAS_STR = "one";
    public static final int N_OUT = 13;

    @Test
    public void testBuildActivationLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put(KERAS_LAYER_PROPERTY_ACTIVATION, ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(KERAS_LAYER_PROPERTY_NAME, LAYER_NAME);

        ActivationLayer layer = buildActivationLayer(config);
        assertEquals(ACTIVATION_DL4J, layer.getActivationFunction());
        assertEquals(LAYER_NAME, layer.getLayerName());
    }

    @Test
    public void testBuildDropoutLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put(KERAS_LAYER_PROPERTY_NAME, LAYER_NAME);
        config.put(KERAS_LAYER_PROPERTY_DROPOUT, DROPOUT_KERAS);

        DropoutLayer layer = buildDropoutLayer(config);
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
    }

    @Test
    public void testBuildDenseLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put(KERAS_LAYER_PROPERTY_ACTIVATION, ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(KERAS_LAYER_PROPERTY_NAME, LAYER_NAME);
        config.put(KERAS_LAYER_PROPERTY_INIT, INIT_KERAS);
        Map<String,Object> W_reg = new HashMap<String,Object>();
        W_reg.put(KERAS_REGULARIZATION_TYPE_L1, L1_REGULARIZATION);
        W_reg.put(KERAS_REGULARIZATION_TYPE_L2, L2_REGULARIZATION);
        config.put(KERAS_LAYER_PROPERTY_W_REGULARIZER, W_reg);
        config.put(KERAS_LAYER_PROPERTY_DROPOUT, DROPOUT_KERAS);
        config.put(KERAS_LAYER_PROPERTY_OUTPUT_DIM, N_OUT);

        DenseLayer layer = buildDenseLayer(config);
        assertEquals(ACTIVATION_DL4J, layer.getActivationFunction());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertEquals(N_OUT, layer.getNOut());
    }

    @Test
    public void testBuildConvolutionLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put(KERAS_LAYER_PROPERTY_ACTIVATION, ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(KERAS_LAYER_PROPERTY_NAME, LAYER_NAME);
        config.put(KERAS_LAYER_PROPERTY_INIT, INIT_KERAS);
        Map<String,Object> W_reg = new HashMap<String,Object>();
        W_reg.put(KERAS_REGULARIZATION_TYPE_L1, L1_REGULARIZATION);
        W_reg.put(KERAS_REGULARIZATION_TYPE_L2, L2_REGULARIZATION);
        config.put(KERAS_LAYER_PROPERTY_W_REGULARIZER, W_reg);
        config.put(KERAS_LAYER_PROPERTY_DROPOUT, DROPOUT_KERAS);
        config.put(KERAS_LAYER_PROPERTY_NB_ROW, KERNEL_SIZE[0]);
        config.put(KERAS_LAYER_PROPERTY_NB_COL, KERNEL_SIZE[1]);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put(KERAS_LAYER_PROPERTY_SUBSAMPLE, subsampleList);
        config.put(KERAS_LAYER_PROPERTY_NB_FILTER, N_OUT);

        ConvolutionLayer layer = buildConvolutionLayer(config);
        assertEquals(ACTIVATION_DL4J, layer.getActivationFunction());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertArrayEquals(KERNEL_SIZE, layer.getKernelSize());
        assertArrayEquals(STRIDE, layer.getStride());
        assertEquals(N_OUT, layer.getNOut());
    }

    @Test
    public void testBuildSubsamplingLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put(KERAS_LAYER_PROPERTY_NAME, LAYER_NAME);
        List<Integer> kernelSizeList = new ArrayList<>();
        kernelSizeList.add(KERNEL_SIZE[0]);
        kernelSizeList.add(KERNEL_SIZE[1]);
        config.put(KERAS_LAYER_PROPERTY_POOL_SIZE, kernelSizeList);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put(KERAS_LAYER_PROPERTY_STRIDES, subsampleList);
        config.put(KERAS_LAYER_PROPERTY_CLASS_NAME, "MaxPooling2D");

        SubsamplingLayer layer = buildSubsamplingLayer(config);
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertArrayEquals(KERNEL_SIZE, layer.getKernelSize());
        assertArrayEquals(STRIDE, layer.getStride());
        assertEquals(POOLING_TYPE, layer.getPoolingType());
    }

    @Test
    public void testBuildGravesLstmLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put(KERAS_LAYER_PROPERTY_ACTIVATION, ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(KERAS_LAYER_PROPERTY_INNER_ACTIVATION, ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(KERAS_LAYER_PROPERTY_NAME, LAYER_NAME);
        config.put(KERAS_LAYER_PROPERTY_INIT, INIT_KERAS);
        config.put(KERAS_LAYER_PROPERTY_INNER_INIT, INIT_KERAS);
        Map<String,Object> W_reg = new HashMap<String,Object>();
        W_reg.put(KERAS_REGULARIZATION_TYPE_L1, L1_REGULARIZATION);
        W_reg.put(KERAS_REGULARIZATION_TYPE_L2, L2_REGULARIZATION);
        config.put(KERAS_LAYER_PROPERTY_W_REGULARIZER, W_reg);
        config.put(KERAS_LAYER_PROPERTY_DROPOUT_W, DROPOUT_KERAS);
        config.put(KERAS_LAYER_PROPERTY_DROPOUT_U, 0.0);
        config.put(KERAS_LAYER_PROPERTY_FORGET_BIAS_INIT, LSTM_FORGET_BIAS_STR);
        config.put(KERAS_LAYER_PROPERTY_OUTPUT_DIM, N_OUT);

        GravesLSTM layer = buildGravesLstmLayer(config);
        assertEquals(ACTIVATION_DL4J, layer.getActivationFunction());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertEquals(LSTM_FORGET_BIAS_DOUBLE, layer.getForgetGateBiasInit(), 0.0);
        assertEquals(N_OUT, layer.getNOut());
    }
}
