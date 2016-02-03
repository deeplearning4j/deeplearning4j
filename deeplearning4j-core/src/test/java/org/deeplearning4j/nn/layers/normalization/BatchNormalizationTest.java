package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 */
public class BatchNormalizationTest {

    @Test
    public void testBatchNormForward() {
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
//                .iterations(5)
//                .seed(123)
//                .list(4)
//                .layer(0, new ConvolutionLayer.Builder().nIn(1).nOut(6).weightInit(WeightInit.XAVIER).activation("relu").build())
//                .layer(1, new BatchNormalization.Builder().nIn(6).build())
//                .layer(2, new DenseLayer.Builder().nIn(6).nOut(2).build())
//                .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//                        .weightInit(WeightInit.XAVIER)
//                        .activation("softmax")
//                        .nIn(2).nOut(10).build())
//                .backprop(true).pretrain(false)
//                .cnnInputSize(28,28,1)
//                .build();
//        MultiLayerNetwork network = new MultiLayerNetwork(conf);
//        network.init();
//        network.fit(next);
//        DataSetIterator iter = new MnistDataSetIterator(5, 5);
//        DataSet next = iter.next();

        org.deeplearning4j.nn.conf.layers.BatchNormalization bN = new org.deeplearning4j.nn.conf.layers.BatchNormalization.Builder().build();
        NeuralNetConfiguration layerConf = new NeuralNetConfiguration.Builder()
                .iterations(1).layer(bN).build();
        Layer layer = LayerFactories.getFactory(layerConf).create(layerConf);

        INDArray data = Nd4j.create(new double[] {
                4.,4.,4.,4.,8.,8.,8.,8.,4.,4.,4.,4.,8.,8.,8.,8.,4.,4.
                ,4.,4.,8.,8.,8.,8.,4.,4.,4.,4.,8.,8.,8.,8,
                2.,2.,2.,2.,4.,4.,4.,4.,2.,2.,2.,2.,4.,4.,4.,4.,
                2.,2.,2.,2.,4.,4.,4.,4.,2.,2.,2.,2.,4.,4.,4.,4.
        },new int[]{2, 2, 4, 4});

        INDArray actualActivation = layer.preOutput(data);

        INDArray expectedOut = Nd4j.create(new double[] {
                1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                -1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,
                -1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,
        },new int[]{2, 2, 4, 4});

        assertEquals(expectedOut, actualActivation);
    }

    @Test
    public void testBatchNormBack(){
        org.deeplearning4j.nn.conf.layers.BatchNormalization bN = new org.deeplearning4j.nn.conf.layers.BatchNormalization.Builder().build();
        NeuralNetConfiguration layerConf = new NeuralNetConfiguration.Builder()
                .iterations(1).layer(bN).build();
        Layer layer = LayerFactories.getFactory(layerConf).create(layerConf);

        INDArray data = Nd4j.create(new double[] {
                4.,4.,4.,4.,8.,8.,8.,8.,4.,4.,4.,4.,8.,8.,8.,8.,4.,4.
                ,4.,4.,8.,8.,8.,8.,4.,4.,4.,4.,8.,8.,8.,8,
                2.,2.,2.,2.,4.,4.,4.,4.,2.,2.,2.,2.,4.,4.,4.,4.,
                2.,2.,2.,2.,4.,4.,4.,4.,2.,2.,2.,2.,4.,4.,4.,4.
        },new int[]{2, 2, 4, 4});
        INDArray epsilon = Nd4j.create(new double[] {
                1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                -1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,
                -1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,
        },new int[]{2, 2, 4, 4});

        INDArray actualActivation = layer.preOutput(data);
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

}
