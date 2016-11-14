package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.*;

import static org.deeplearning4j.nn.modelimport.keras.LayerConfiguration.*;
import static org.junit.Assert.*;

/**
 * Created by davekale on 10/28/16.
 */
public class LayerBuildTest {
    String activation = "identity";
    String name = "test_layer";
    WeightInit init = WeightInit.UNIFORM;
    double l1 = 0.01;
    double l2 = 0.02;
    double dropout = 0.3;
    int[] kernelSize = new int[]{1, 2};
    int[] stride = new int[]{3, 4};
    SubsamplingLayer.PoolingType poolType = SubsamplingLayer.PoolingType.MAX;
    double forgetBiasDouble = 1.0;
    String forgetBiasStr = "one";
    int nOut = 13;

    @Test
    public void testBuildDenseLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put("activation", "linear"); // keras linear -> dl4j identity
        config.put("name", name);
        config.put("init", "uniform");
        Map<String,Object> W_reg = new HashMap<String,Object>();
        W_reg.put("l1", l1);
        W_reg.put("l2", l2);
        config.put("W_regularizer", W_reg);
        config.put("dropout", dropout);
        config.put("output_dim", nOut);

        DenseLayer layer = buildDenseLayer(config);
        assertEquals(activation, layer.getActivationFunction());
        assertEquals(name, layer.getLayerName());
        assertEquals(init, layer.getWeightInit());
        assertEquals(l1, layer.getL1(), 0.0);
        assertEquals(l2, layer.getL2(), 0.0);
        assertEquals(dropout, layer.getDropOut(), 0.0);
        assertEquals(nOut, layer.getNOut());
    }

    @Test
    public void testBuildConvolutionLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put("activation", "linear"); // keras linear -> dl4j identity
        config.put("name", name);
        config.put("init", "uniform");
        Map<String,Object> W_reg = new HashMap<String,Object>();
        W_reg.put("l1", l1);
        W_reg.put("l2", l2);
        config.put("W_regularizer", W_reg);
        config.put("dropout", dropout);
        config.put("nb_row", kernelSize[0]);
        config.put("nb_col", kernelSize[1]);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(stride[0]);
        subsampleList.add(stride[1]);
        config.put("subsample", subsampleList);
        config.put("nb_filter", nOut);

        ConvolutionLayer layer = buildConvolutionLayer(config);
        assertEquals(activation, layer.getActivationFunction());
        assertEquals(name, layer.getLayerName());
        assertEquals(init, layer.getWeightInit());
        assertEquals(l1, layer.getL1(), 0.0);
        assertEquals(l2, layer.getL2(), 0.0);
        assertEquals(dropout, layer.getDropOut(), 0.0);
        assertArrayEquals(kernelSize, layer.getKernelSize());
        assertArrayEquals(stride, layer.getStride());
        assertEquals(nOut, layer.getNOut());
    }

    @Test
    public void testBuildSubsamplingLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put("name", name);
        List<Integer> kernelSizeList = new ArrayList<>();
        kernelSizeList.add(kernelSize[0]);
        kernelSizeList.add(kernelSize[1]);
        config.put("pool_size", kernelSizeList);
        List<Integer> subsampleList = new ArrayList<>();
        subsampleList.add(stride[0]);
        subsampleList.add(stride[1]);
        config.put("strides", subsampleList);
        config.put("keras_class", "MaxPooling2D");

        SubsamplingLayer layer = buildSubsamplingLayer(config);
        assertEquals(name, layer.getLayerName());
        assertArrayEquals(kernelSize, layer.getKernelSize());
        assertArrayEquals(stride, layer.getStride());
        assertEquals(poolType, layer.getPoolingType());
    }

    @Test
    public void testBuildGravesLstmLayer() throws Exception {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put("activation", "linear"); // keras linear -> dl4j identity
        config.put("inner_activation", "linear"); // keras linear -> dl4j identity
        config.put("name", name);
        config.put("init", "uniform");
        config.put("inner_init", "uniform");
        Map<String,Object> W_reg = new HashMap<String,Object>();
        W_reg.put("l1", l1);
        W_reg.put("l2", l2);
        config.put("W_regularizer", W_reg);
        config.put("dropout_W", dropout);
        config.put("dropout_U", 0.0);
        config.put("forget_bias_init", forgetBiasStr);
        config.put("output_dim", nOut);

        GravesLSTM layer = buildGravesLstmLayer(config);
        assertEquals(activation, layer.getActivationFunction());
        assertEquals(name, layer.getLayerName());
        assertEquals(init, layer.getWeightInit());
        assertEquals(l1, layer.getL1(), 0.0);
        assertEquals(l2, layer.getL2(), 0.0);
        assertEquals(dropout, layer.getDropOut(), 0.0);
        assertEquals(forgetBiasDouble, layer.getForgetGateBiasInit(), 0.0);
        assertEquals(nOut, layer.getNOut());
    }
}
