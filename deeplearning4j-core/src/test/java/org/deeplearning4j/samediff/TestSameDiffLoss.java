package org.deeplearning4j.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.samediff.testlayers.SameDiffLoss;
import org.deeplearning4j.samediff.testlayers.SameDiffOutput;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

@Slf4j
public class TestSameDiffLoss {

    @Test
    public void testSameDiffLossBasic() {

        int minibatch = 3;
        int nIn = 3;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder().nIn(3).nOut(4).activation(Activation.TANH).build())
                .layer(new SameDiffLoss.Builder().lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray in = Nd4j.rand(minibatch, nIn);
        INDArray out = net.output(in);
        assertArrayEquals(new int[]{minibatch, nOut}, out.shape());

        INDArray label = Nd4j.rand(minibatch, nOut);
        net.setLabels(label);
        net.computeGradientAndScore();
        double score = net.score();
        assertTrue(score > 0);
    }

    @Test
    public void testSameDiffLossVsDl4j() {

        for (int minibatch : new int[]{5, 1}) {
            int nIn = 3;
            int nOut = 4;

            LossFunctions.LossFunction[] lossFns = new LossFunctions.LossFunction[]{
                    LossFunctions.LossFunction.MSE,
//                    LossFunctions.LossFunction.MCXENT,
//                    LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD,
//                    LossFunctions.LossFunction.L2,
//                    LossFunctions.LossFunction.SQUARED_LOSS,
//                    LossFunctions.LossFunction.KL_DIVERGENCE,
//                    LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR,
//                    LossFunctions.LossFunction.XENT,
//                    LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR
            };

            for (int i = 0; i < lossFns.length; i++) {
                LossFunctions.LossFunction lf = lossFns[i];
                log.info("Starting test - " + lf);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).activation(Activation.TANH).build())
                        .layer(new SameDiffLoss.Builder()
                                .lossFunction(lf)
                                .build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                assertNotNull(net.paramTable());

                MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).activation(Activation.TANH).build())
                        .layer(new LossLayer.Builder()
                                .lossFunction(lf)
                                .activation(Activation.IDENTITY)
                                .build())
                        .build();

                MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
                net2.init();

                net.params().assign(net2.params());

                //Check params:
                assertEquals(net2.params(), net.params());
                Map<String, INDArray> params1 = net.paramTable();
                Map<String, INDArray> params2 = net2.paramTable();
                assertEquals(params2, params1);

                INDArray in = Nd4j.rand(minibatch, nIn);
                INDArray out = net.output(in);
                INDArray outExp = net2.output(in);

                assertEquals(outExp, out);

                //Check scores:
                INDArray label = Nd4j.rand(minibatch, nOut);
                net.setLabels(label);
                net2.setLabels(label);

                net.computeGradientAndScore();
                net2.computeGradientAndScore();

                double scoreExp = net2.score();
                double scoreAct = net.score();
                assertTrue(scoreExp > 0);
                assertEquals(scoreExp, scoreAct, 1e-6);

                INDArray gradExp = net2.getFlattenedGradients();
                INDArray gradAct = net.getFlattenedGradients();

                assertEquals(gradExp, gradAct);

                //Also check serialization:
                MultiLayerNetwork netLoaded = TestUtils.testModelSerialization(net);
                INDArray outLoaded = netLoaded.output(in);

                assertEquals(outExp, outLoaded);
            }
        }
    }
}
