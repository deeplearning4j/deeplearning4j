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
public class LayerBuildTest {
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
        config.put("activation", ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put("name", LAYER_NAME);

        ActivationLayer layer = buildActivationLayer(config);
        assertEquals(ACTIVATION_DL4J, layer.getActivationFunction());
        assertEquals(LAYER_NAME, layer.getLayerName());
    }

    @Test
    public void testBuildDropoutLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put("name", LAYER_NAME);
        config.put("dropout", DROPOUT_KERAS);

        DropoutLayer layer = buildDropoutLayer(config);
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
    }

    @Test
    public void testBuildDenseLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put("activation", ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put("name", LAYER_NAME);
        config.put("init", INIT_KERAS);
        Map<String,Object> W_reg = new HashMap<String,Object>();
        W_reg.put("l1", L1_REGULARIZATION);
        W_reg.put("l2", L2_REGULARIZATION);
        config.put("W_regularizer", W_reg);
        config.put("dropout", DROPOUT_KERAS);
        config.put("output_dim", N_OUT);

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
        config.put("activation", ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put("name", LAYER_NAME);
        config.put("init", INIT_KERAS);
        Map<String,Object> W_reg = new HashMap<String,Object>();
        W_reg.put("l1", L1_REGULARIZATION);
        W_reg.put("l2", L2_REGULARIZATION);
        config.put("W_regularizer", W_reg);
        config.put("dropout", DROPOUT_KERAS);
        config.put("nb_row", KERNEL_SIZE[0]);
        config.put("nb_col", KERNEL_SIZE[1]);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put("subsample", subsampleList);
        config.put("nb_filter", N_OUT);

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
        config.put("name", LAYER_NAME);
        List<Integer> kernelSizeList = new ArrayList<>();
        kernelSizeList.add(KERNEL_SIZE[0]);
        kernelSizeList.add(KERNEL_SIZE[1]);
        config.put("pool_size", kernelSizeList);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put("strides", subsampleList);
        config.put("keras_class", "MaxPooling2D");

        SubsamplingLayer layer = buildSubsamplingLayer(config);
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertArrayEquals(KERNEL_SIZE, layer.getKernelSize());
        assertArrayEquals(STRIDE, layer.getStride());
        assertEquals(POOLING_TYPE, layer.getPoolingType());
    }

    @Test
    public void testBuildGravesLstmLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put("activation", ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put("inner_activation", ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put("name", LAYER_NAME);
        config.put("init", INIT_KERAS);
        config.put("inner_init", INIT_KERAS);
        Map<String,Object> W_reg = new HashMap<String,Object>();
        W_reg.put("l1", L1_REGULARIZATION);
        W_reg.put("l2", L2_REGULARIZATION);
        config.put("W_regularizer", W_reg);
        config.put("dropout_W", DROPOUT_KERAS);
        config.put("dropout_U", 0.0);
        config.put("forget_bias_init", LSTM_FORGET_BIAS_STR);
        config.put("output_dim", N_OUT);

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
