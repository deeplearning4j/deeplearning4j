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
import org.nd4j.linalg.dataset.DataSet;
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
    public void testReductionsBackwards() {

        for (int i = 1; i < 7; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 3;
            SDVariable input = sd.var("in", new int[]{-1, nOut});
            SDVariable label = sd.var("label", new int[]{-1, nOut});

            SDVariable diff = input.sub(label);
            SDVariable sqDiff = diff.mul(diff);
            SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);

            SDVariable loss;
            String name;
            switch (i) {
                case 0:
                    loss = sd.mean("loss", msePerEx, 0);
                    name = "mean";
                    break;
                case 1:
                    loss = sd.sum("loss", msePerEx, 0);
                    name = "sum";
                    break;
                case 2:
                    loss = sd.standardDeviation("loss", msePerEx, true, 0);
                    name = "stdev";
                    break;
                case 3:
                    loss = sd.min("loss", msePerEx, 0);
                    name = "min";
                    break;
                case 4:
                    loss = sd.max("loss", msePerEx, 0);
                    name = "max";
                    break;
                case 5:
                    loss = sd.variance("loss", msePerEx, true, 0);
                    name = "variance";
                    break;
                case 6:
                    loss = sd.prod("loss", msePerEx, 0);
                    name = "prod";
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            INDArray inputArr = Nd4j.rand(minibatch, nOut);
            INDArray labelArr = Nd4j.rand(minibatch, nOut);

            sd.associateArrayWithVariable(inputArr, input);
            sd.associateArrayWithVariable(labelArr, label);

            INDArray result = sd.execAndEndResult();
            assertEquals(1, result.length());

            Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> p = sd.execBackwards();
        }
    }

    @Test
    public void testReductionsBackwards2() {

        for (int i = 0; i < 7; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 3;
            SDVariable input = sd.var("in", new int[]{-1, nOut});
            SDVariable label = sd.var("label", new int[]{-1, nOut});

            SDVariable diff = input.sub(label);
            SDVariable sqDiff = diff.mul(diff);
            SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);

            SDVariable loss;
            String name;
            switch (i) {
                case 0:
                    loss = sd.mean("loss", msePerEx);
                    name = "mean";
                    break;
                case 1:
                    loss = sd.sum("loss", msePerEx);
                    name = "sum";
                    break;
                case 2:
                    loss = sd.standardDeviation("loss", msePerEx, true);
                    name = "stdev";
                    break;
                case 3:
                    loss = sd.min("loss", msePerEx);
                    name = "min";
                    break;
                case 4:
                    loss = sd.max("loss", msePerEx);
                    name = "max";
                    break;
                case 5:
                    loss = sd.variance("loss", msePerEx, true);
                    name = "variance";
                    break;
                case 6:
                    loss = sd.prod("loss", msePerEx);
                    name = "prod";
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            INDArray inputArr = Nd4j.rand(minibatch, nOut);
            INDArray labelArr = Nd4j.rand(minibatch, nOut);

            sd.associateArrayWithVariable(inputArr, input);
            sd.associateArrayWithVariable(labelArr, label);

            INDArray result = sd.execAndEndResult();
            assertEquals(1, result.length());

            Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> p = sd.execBackwards();
        }
    }

    @Test
    public void testSameDiffLossVsDl4j() {

        double[] l1s = new double[]{0.0, 0.0, 0.4, 0.4};
        double[] l2s = new double[]{0.0, 0.3, 0.0, 0.3};

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

                for( int j=0; j<l1s.length; j++ ) {
                    double l1 = l1s[j];
                    double l2 = l2s[j];

                    LossFunctions.LossFunction lf = lossFns[i];
                    String msg = "Starting test - " + lf + ", minibatch=" + minibatch + ", l1=" + l1 + ", l2=" + l2;
                    log.info(msg);

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .list()
                            .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).activation(Activation.TANH).build())
                            .layer(new SameDiffLoss.Builder()
                                    .lossFunction(lf)
                                    .build())
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    assertNotNull(msg, net.paramTable());

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
                    assertEquals(msg, net2.params(), net.params());
                    Map<String, INDArray> params1 = net.paramTable();
                    Map<String, INDArray> params2 = net2.paramTable();
                    assertEquals(msg, params2, params1);

                    INDArray in = Nd4j.rand(minibatch, nIn);
                    INDArray out = net.output(in);
                    INDArray outExp = net2.output(in);

                    assertEquals(msg, outExp, out);

                    //Check scores:
                    INDArray label = Nd4j.rand(minibatch, nOut);
                    net.setLabels(label);
                    net2.setLabels(label);

                    net.computeGradientAndScore();
                    net2.computeGradientAndScore();

                    double scoreExp = net2.score();
                    double scoreAct = net.score();
                    assertTrue(msg, scoreExp > 0);
                    assertEquals(msg, scoreExp, scoreAct, 1e-6);

                    //Test: computeScoreForExamples
                    for(boolean includeReg : new boolean[]{true, false}) {
                        INDArray expScoreForEx = net2.scoreExamples(new DataSet(in, label), true);
                        INDArray actScoreForEx = net.scoreExamples(new DataSet(in, label), true);

                        String msg2 = msg + ", addRegTerms=" + includeReg;
                        assertEquals(msg2, expScoreForEx, actScoreForEx);
                    }

                    //TODO GRADIENTS NEED FIXING - maybe related: https://github.com/deeplearning4j/nd4j/issues/2485
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
}
