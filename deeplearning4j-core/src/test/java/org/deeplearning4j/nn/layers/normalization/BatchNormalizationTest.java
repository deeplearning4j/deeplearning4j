package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 */
public class BatchNormalizationTest {
    protected INDArray input = Nd4j.create(new double[] {
            4.,4.,4.,4.,8.,8.,8.,8.,4.,4.,4.,4.,8.,8.,8.,8.,4.,4.
            ,4.,4.,8.,8.,8.,8.,4.,4.,4.,4.,8.,8.,8.,8,
            2.,2.,2.,2.,4.,4.,4.,4.,2.,2.,2.,2.,4.,4.,4.,4.,
            2.,2.,2.,2.,4.,4.,4.,4.,2.,2.,2.,2.,4.,4.,4.,4.
    },new int[]{2, 2, 4, 4});


    protected INDArray epsilon = Nd4j.create(new double[] {
            1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
            1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
            -1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,
            -1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,
    },new int[]{2, 2, 4, 4});

    protected Layer layer;
    protected INDArray activationsActual;

    @Before
    public void doBefore() {
        BatchNormalization bN = new BatchNormalization.Builder().build();
        NeuralNetConfiguration layerConf = new NeuralNetConfiguration.Builder()
                .iterations(1).layer(bN).build();
        layer = LayerFactories.getFactory(layerConf).create(layerConf);
        activationsActual = layer.preOutput(input);
    }

    @Test
    public void testBatchNormForward() {

        INDArray activationsExpected = epsilon;
        assertEquals(activationsExpected, activationsActual);
        assertArrayEquals(activationsExpected.shape(), activationsActual.shape());
    }

    @Test
    public void testBatchNormBack(){
        Pair<Gradient, INDArray> actualOut = layer.backpropGradient(epsilon);

        INDArray expectedEpsilon = Nd4j.create(new double[] {
                1.,1.,1.,1.,.5,.5,.5,.5,1.,1.,1.,1.,.5,.5,.5,.5,
                1.,1.,1.,1.,.5,.5,.5,.5,1.,1.,1.,1.,.5,.5,.5,.5,
                -1.,-1.,-1.,-1.,-.5,-.5,-.5,-.5,-1.,-1.,-1.,-1.,-.5,-.5,-.5,-.5,
                -1.,-1.,-1.,-1.,-.5,-.5,-.5,-.5,-1.,-1.,-1.,-1.,-.5,-.5,-.5,-.5,
        },new int[]{2, 2, 4, 4});

        assertEquals(expectedEpsilon, actualOut.getSecond());
        assertEquals(null, actualOut.getFirst().getGradientFor("W"));
    }

    @Test
    public void testMultiCNNLayer() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(1)
                .seed(123)
                .list()
                .layer(0, new ConvolutionLayer.Builder().nIn(1).nOut(6).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new DenseLayer.Builder().nOut(2).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(10).build())
                .backprop(true).pretrain(false)
                .cnnInputSize(28,28,1)
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        network.fit(next);

    }

}
