package org.deeplearning4j.nn.dtypes;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

public class DTypeTests extends BaseDL4JTest {

    @Test
    public void testMultiLayerNetworkTypeConversion(){

        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                .layer(new DenseLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray inD = Nd4j.rand(DataType.DOUBLE, 1, 10);
        INDArray lD = Nd4j.create(DataType.DOUBLE, 1,10);
        net.fit(inD, lD);

        INDArray outDouble = net.output(inD);
        net.setInput(inD);
        net.setLabels(lD);
        net.computeGradientAndScore();
        double scoreDouble = net.score();
        INDArray grads = net.getFlattenedGradients();
        INDArray u = net.getUpdater().getStateViewArray();

        MultiLayerNetwork netFloat = net.convertDataType(DataType.FLOAT);
        netFloat.initGradientsView();
        assertEquals(DataType.FLOAT, netFloat.params().dataType());
        assertEquals(DataType.FLOAT, netFloat.getFlattenedGradients().dataType());
        assertEquals(DataType.FLOAT, netFloat.getUpdater(true).getStateViewArray().dataType());
        INDArray inF = inD.castTo(DataType.FLOAT);
        INDArray lF = lD.castTo(DataType.FLOAT);
        INDArray outFloat = netFloat.output(inF);
        netFloat.setInput(inF);
        netFloat.setLabels(lF);
        netFloat.computeGradientAndScore();
        double scoreFloat = netFloat.score();
        INDArray gradsFloat = netFloat.getFlattenedGradients();
        INDArray uFloat = net.getUpdater().getStateViewArray();

        assertEquals(scoreDouble, scoreFloat, 1e-6);
        assertEquals(outDouble.castTo(DataType.FLOAT), outFloat);
        assertEquals(grads.castTo(DataType.FLOAT), gradsFloat);
        assertEquals(u.castTo(DataType.FLOAT), uFloat);


        MultiLayerNetwork netFP16 = net.convertDataType(DataType.HALF);
        netFP16.initGradientsView();
        assertEquals(DataType.HALF, netFP16.params().dataType());
        assertEquals(DataType.HALF, netFP16.getFlattenedGradients().dataType());
        assertEquals(DataType.HALF, netFP16.getUpdater(true).getStateViewArray().dataType());
    }

}
