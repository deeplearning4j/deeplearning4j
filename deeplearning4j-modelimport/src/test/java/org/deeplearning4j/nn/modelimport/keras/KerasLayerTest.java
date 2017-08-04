package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.modelimport.keras.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.*;

import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.*;
import static org.deeplearning4j.nn.modelimport.keras.layers.KerasLstm.*;
import static org.deeplearning4j.nn.modelimport.keras.layers.KerasBatchNormalization.*;
import static org.junit.Assert.*;

/**
 * Unit tests for end-to-end Keras layer configuration import.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasLayerTest {
    public static final String ACTIVATION_KERAS = "linear";
    public static final String INNER_ACTIVATION_KERAS = "hard_sigmoid";
    public static final String ACTIVATION_DL4J = "identity";
    public static final String LAYER_NAME = "test_layer";
    public static final String INIT_KERAS = "glorot_normal";
    public static final WeightInit INIT_DL4J = WeightInit.XAVIER;
    public static final double L1_REGULARIZATION = 0.01;
    public static final double L2_REGULARIZATION = 0.02;
    public static final double DROPOUT_KERAS = 0.3;
    public static final double DROPOUT_DL4J = 1 - DROPOUT_KERAS;
    public static final int[] KERNEL_SIZE = new int[] {1, 2};
    public static final int[] STRIDE = new int[] {3, 4};
    public static final PoolingType POOLING_TYPE = PoolingType.MAX;
    public static final double LSTM_FORGET_BIAS_DOUBLE = 1.0;
    public static final String LSTM_FORGET_BIAS_STR = "one";
    public static final boolean LSTM_UNROLL = true;
    public static final int N_OUT = 13;
    public static final String BORDER_MODE_VALID = "valid";
    public static final int[] VALID_PADDING = new int[] {0, 0};
    public static final String LAYER_CLASS_NAME_BATCHNORMALIZATION = "BatchNormalization";
    public static final double EPSILON = 1E-5;
    public static final double MOMENTUM = 0.99;

    @Test
    public void testBuildActivationLayer() throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(LAYER_FIELD_CLASS_NAME, LAYER_CLASS_NAME_ACTIVATION);
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(LAYER_FIELD_ACTIVATION, ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(LAYER_FIELD_NAME, LAYER_NAME);
        layerConfig.put(LAYER_FIELD_CONFIG, config);

        ActivationLayer layer = new KerasActivation(layerConfig).getActivationLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
    }

    @Test
    public void testBuildDropoutLayer() throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(LAYER_FIELD_CLASS_NAME, LAYER_CLASS_NAME_DROPOUT);
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(LAYER_FIELD_NAME, LAYER_NAME);
        config.put(LAYER_FIELD_DROPOUT, DROPOUT_KERAS);
        layerConfig.put(LAYER_FIELD_CONFIG, config);

        DropoutLayer layer = new KerasDropout(layerConfig).getDropoutLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
    }

    @Test
    public void testBuildDenseLayer() throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(LAYER_FIELD_CLASS_NAME, LAYER_CLASS_NAME_DENSE);
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(LAYER_FIELD_ACTIVATION, ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(LAYER_FIELD_NAME, LAYER_NAME);
        config.put(LAYER_FIELD_INIT, INIT_KERAS);
        Map<String, Object> W_reg = new HashMap<String, Object>();
        W_reg.put(REGULARIZATION_TYPE_L1, L1_REGULARIZATION);
        W_reg.put(REGULARIZATION_TYPE_L2, L2_REGULARIZATION);
        config.put(LAYER_FIELD_W_REGULARIZER, W_reg);
        config.put(LAYER_FIELD_DROPOUT, DROPOUT_KERAS);
        config.put(LAYER_FIELD_OUTPUT_DIM, N_OUT);
        layerConfig.put(LAYER_FIELD_CONFIG, config);

        DenseLayer layer = new KerasDense(layerConfig, false).getDenseLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertEquals(N_OUT, layer.getNOut());
    }

    @Test
    public void testBuildConvolutionLayer() throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(LAYER_FIELD_CLASS_NAME, LAYER_CLASS_NAME_CONVOLUTION_2D);
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(LAYER_FIELD_ACTIVATION, ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(LAYER_FIELD_NAME, LAYER_NAME);
        config.put(LAYER_FIELD_INIT, INIT_KERAS);
        Map<String, Object> W_reg = new HashMap<String, Object>();
        W_reg.put(REGULARIZATION_TYPE_L1, L1_REGULARIZATION);
        W_reg.put(REGULARIZATION_TYPE_L2, L2_REGULARIZATION);
        config.put(LAYER_FIELD_W_REGULARIZER, W_reg);
        config.put(LAYER_FIELD_DROPOUT, DROPOUT_KERAS);
        config.put(LAYER_FIELD_NB_ROW, KERNEL_SIZE[0]);
        config.put(LAYER_FIELD_NB_COL, KERNEL_SIZE[1]);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put(LAYER_FIELD_SUBSAMPLE, subsampleList);
        config.put(LAYER_FIELD_NB_FILTER, N_OUT);
        config.put(LAYER_FIELD_BORDER_MODE, BORDER_MODE_VALID);
        layerConfig.put(LAYER_FIELD_CONFIG, config);

        ConvolutionLayer layer = new KerasConvolution(layerConfig).getConvolutionLayer();
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
    }

    @Test
    public void testBuildSubsamplingLayer() throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(LAYER_FIELD_CLASS_NAME, LAYER_CLASS_NAME_MAX_POOLING_2D);
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(LAYER_FIELD_NAME, LAYER_NAME);
        List<Integer> kernelSizeList = new ArrayList<>();
        kernelSizeList.add(KERNEL_SIZE[0]);
        kernelSizeList.add(KERNEL_SIZE[1]);
        config.put(LAYER_FIELD_POOL_SIZE, kernelSizeList);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(STRIDE[0]);
        subsampleList.add(STRIDE[1]);
        config.put(LAYER_FIELD_STRIDES, subsampleList);
        config.put(LAYER_FIELD_BORDER_MODE, BORDER_MODE_VALID);
        layerConfig.put(LAYER_FIELD_CONFIG, config);

        SubsamplingLayer layer = new KerasPooling(layerConfig).getSubsamplingLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertArrayEquals(KERNEL_SIZE, layer.getKernelSize());
        assertArrayEquals(STRIDE, layer.getStride());
        assertEquals(POOLING_TYPE, layer.getPoolingType());
        assertEquals(ConvolutionMode.Truncate, layer.getConvolutionMode());
        assertArrayEquals(VALID_PADDING, layer.getPadding());
    }

    @Test
    public void testBuildGravesLstmLayer() throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(LAYER_FIELD_CLASS_NAME, LAYER_CLASS_NAME_LSTM);
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(LAYER_FIELD_ACTIVATION, ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(LAYER_FIELD_INNER_ACTIVATION, INNER_ACTIVATION_KERAS); // keras linear -> dl4j identity
        config.put(LAYER_FIELD_NAME, LAYER_NAME);
        config.put(LAYER_FIELD_INIT, INIT_KERAS);
        config.put(LAYER_FIELD_INNER_INIT, INIT_KERAS);
        Map<String, Object> W_reg = new HashMap<String, Object>();
        W_reg.put(REGULARIZATION_TYPE_L1, L1_REGULARIZATION);
        W_reg.put(REGULARIZATION_TYPE_L2, L2_REGULARIZATION);
        config.put(LAYER_FIELD_W_REGULARIZER, W_reg);
        config.put(LAYER_FIELD_DROPOUT_W, DROPOUT_KERAS);
        config.put(LAYER_FIELD_DROPOUT_U, 0.0);
        config.put(LAYER_FIELD_FORGET_BIAS_INIT, LSTM_FORGET_BIAS_STR);
        config.put(LAYER_FIELD_OUTPUT_DIM, N_OUT);
        config.put(LAYER_FIELD_UNROLL, LSTM_UNROLL);
        layerConfig.put(LAYER_FIELD_CONFIG, config);

        GravesLSTM layer = new KerasLstm(layerConfig).getGravesLSTMLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInit());
        assertEquals(L1_REGULARIZATION, layer.getL1(), 0.0);
        assertEquals(L2_REGULARIZATION, layer.getL2(), 0.0);
        assertEquals(DROPOUT_DL4J, layer.getDropOut(), 0.0);
        assertEquals(LSTM_FORGET_BIAS_DOUBLE, layer.getForgetGateBiasInit(), 0.0);
        assertEquals(N_OUT, layer.getNOut());
    }

    @Test
    public void testBuildBatchNormalizationLayer() throws Exception {
        Map<String, Object> layerConfig = new HashMap<String, Object>();
        layerConfig.put(LAYER_FIELD_CLASS_NAME, LAYER_CLASS_NAME_BATCHNORMALIZATION);
        Map<String, Object> config = new HashMap<String, Object>();
        config.put(LAYER_FIELD_NAME, LAYER_NAME);
        config.put(LAYER_FIELD_EPSILON, EPSILON);
        config.put(LAYER_FIELD_MOMENTUM, MOMENTUM);
        config.put(LAYER_FIELD_GAMMA_REGULARIZER, null);
        config.put(LAYER_FIELD_BETA_REGULARIZER, null);
        config.put(LAYER_FIELD_MODE, 0);
        config.put(LAYER_FIELD_AXIS, 3);
        layerConfig.put(LAYER_FIELD_CONFIG, config);

        BatchNormalization layer = new KerasBatchNormalization(layerConfig).getBatchNormalizationLayer();
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(EPSILON, layer.getEps(), 0.0);
        assertEquals(MOMENTUM, layer.getMomentum(), 0.0);
    }
}
