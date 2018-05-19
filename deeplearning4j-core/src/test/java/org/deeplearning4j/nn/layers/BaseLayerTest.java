package org.deeplearning4j.nn.layers;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Created by nyghtowl on 11/15/15.
 */
public class BaseLayerTest extends BaseDL4JTest {

    protected INDArray weight = Nd4j.create(new double[] {0.10, -0.20, -0.15, 0.05}, new int[] {2, 2});
    protected INDArray bias = Nd4j.create(new double[] {0.5, 0.5}, new int[] {1, 2});
    protected Map<String, INDArray> paramTable;

    @Before
    public void doBefore() {
        paramTable = new HashMap<>();
        paramTable.put("W", weight);
        paramTable.put("b", bias);

    }

    @Test
    public void testSetExistingParamsConvolutionSingleLayer() {
        Layer layer = configureSingleLayer();
        assertNotEquals(paramTable, layer.paramTable());

        layer.setParamTable(paramTable);
        assertEquals(paramTable, layer.paramTable());
    }


    @Test
    public void testSetExistingParamsDenseMultiLayer() {
        MultiLayerNetwork net = configureMultiLayer();

        for (Layer layer : net.getLayers()) {
            assertNotEquals(paramTable, layer.paramTable());
            layer.setParamTable(paramTable);
            assertEquals(paramTable, layer.paramTable());
        }
    }


    public Layer configureSingleLayer() {
        int nIn = 2;
        int nOut = 2;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .layer(new ConvolutionLayer.Builder().nIn(nIn).nOut(nOut).build()).build();

        val numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        return conf.getLayer().instantiate(conf, null, 0, params, true);
    }


    public MultiLayerNetwork configureMultiLayer() {
        int nIn = 2;
        int nOut = 2;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
                        .layer(1, new OutputLayer.Builder().nIn(nIn).nOut(nOut).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

}
