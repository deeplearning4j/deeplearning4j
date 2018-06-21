package org.deeplearning4j.nn.layers.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.samediff.testlayers.SameDiffDense;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

@Slf4j
public class TestSameDiffDense {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    @Test
    public void testSameDiffDenseBasic() {
        //Only run test for CPU backend for now:
        assumeTrue("CPU".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend")));

        int nIn = 3;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Map<String, INDArray> pt1 = net.getLayer(0).paramTable();
        assertNotNull(pt1);
        assertEquals(2, pt1.size());
        assertNotNull(pt1.get(DefaultParamInitializer.WEIGHT_KEY));
        assertNotNull(pt1.get(DefaultParamInitializer.BIAS_KEY));

        assertArrayEquals(new long[]{nIn, nOut}, pt1.get(DefaultParamInitializer.WEIGHT_KEY).shape());
        assertArrayEquals(new long[]{1, nOut}, pt1.get(DefaultParamInitializer.BIAS_KEY).shape());
    }

    @Test
    public void testSameDiffDenseForward() {
        //Only run test for CPU backend for now:
        assumeTrue("CPU".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend")));

        for (int minibatch : new int[]{5, 1}) {
            int nIn = 3;
            int nOut = 4;

            Activation[] afns = new Activation[]{
                    Activation.TANH,
                    Activation.SIGMOID,
                    Activation.ELU,
                    Activation.IDENTITY,
                    Activation.SOFTPLUS,
                    Activation.SOFTSIGN,
                    Activation.CUBE,
                    Activation.HARDTANH,
                    Activation.RELU
            };

            for (Activation a : afns) {
                log.info("Starting test - " + a);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut)
                                .activation(a)
                                .build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                assertNotNull(net.paramTable());

                MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(new DenseLayer.Builder().activation(a).nIn(nIn).nOut(nOut).build())
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

                //Also check serialization:
                MultiLayerNetwork netLoaded = TestUtils.testModelSerialization(net);
                INDArray outLoaded = netLoaded.output(in);

                assertEquals(outExp, outLoaded);
            }
        }
    }

    @Test
    public void testSameDiffDenseForwardMultiLayer() {
        //Only run test for CPU backend for now:
        assumeTrue("CPU".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend")));

        for (int minibatch : new int[]{5, 1}) {
            int nIn = 3;
            int nOut = 4;

            Activation[] afns = new Activation[]{
                    Activation.TANH,
                    Activation.SIGMOID,
                    Activation.ELU,
                    Activation.IDENTITY,
                    Activation.SOFTPLUS,
                    Activation.SOFTSIGN,
                    Activation.CUBE,    //https://github.com/deeplearning4j/nd4j/issues/2426
                    Activation.HARDTANH,
                    Activation.RELU      //JVM crash
            };

            for (Activation a : afns) {
                log.info("Starting test - " + a);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .list()
                        .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut)
                                .weightInit(WeightInit.XAVIER)
                                .activation(a).build())
                        .layer(new SameDiffDense.Builder().nIn(nOut).nOut(nOut)
                                .weightInit(WeightInit.XAVIER)
                                .activation(a).build())
                        .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut)
                                .weightInit(WeightInit.XAVIER)
                                .activation(a).build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                assertNotNull(net.paramTable());

                MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(new DenseLayer.Builder().activation(a).nIn(nIn).nOut(nOut).build())
                        .layer(new DenseLayer.Builder().activation(a).nIn(nOut).nOut(nOut).build())
                        .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut)
                                .activation(a).build())
                        .build();

                MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
                net2.init();

//                net.params().assign(net2.params());
                assertEquals(net2.params(), net.params());

                //Check params:
                assertEquals(net2.params(), net.params());
                Map<String, INDArray> params1 = net.paramTable();
                Map<String, INDArray> params2 = net2.paramTable();
                assertEquals(params2, params1);

                INDArray in = Nd4j.rand(minibatch, nIn);
                INDArray out = net.output(in);
                INDArray outExp = net2.output(in);

                assertEquals(outExp, out);

                //Also check serialization:
                MultiLayerNetwork netLoaded = TestUtils.testModelSerialization(net);
                INDArray outLoaded = netLoaded.output(in);

                assertEquals(outExp, outLoaded);
            }
        }
    }

    @Test   //(expected = UnsupportedOperationException.class)   //Backprop not yet supported
    public void testSameDiffDenseBackward() {

        int nIn = 3;
        int nOut = 4;

        for(boolean workspaces : new boolean[]{false, true}) {

            for (int minibatch : new int[]{5, 1}) {

                Activation[] afns = new Activation[]{
                        Activation.TANH,
                        Activation.SIGMOID,
                        Activation.ELU, Activation.IDENTITY, Activation.SOFTPLUS, Activation.SOFTSIGN,
                        Activation.HARDTANH,
                        Activation.CUBE,    //https://github.com/deeplearning4j/nd4j/issues/2426
                        Activation.RELU      //JVM crash
                };

                for (Activation a : afns) {
                    log.info("Starting test - " + a + " - minibatch " + minibatch + ", workspaces: " + workspaces);
                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .trainingWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                            .inferenceWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                            .list()
                            .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut)
                                    .activation(a)
                                    .build())
                            .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut).activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                            .build();

                    MultiLayerNetwork netSD = new MultiLayerNetwork(conf);
                    netSD.init();

                    MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                            .list()
                            .layer(new DenseLayer.Builder().activation(a).nIn(nIn).nOut(nOut).build())
                            .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut).activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                            .build();

                    MultiLayerNetwork netStandard = new MultiLayerNetwork(conf2);
                    netStandard.init();

                    netSD.params().assign(netStandard.params());

                    //Check params:
                    assertEquals(netStandard.params(), netSD.params());
                    assertEquals(netStandard.paramTable(), netSD.paramTable());

                    INDArray in = Nd4j.rand(minibatch, nIn);
                    INDArray l = TestUtils.randomOneHot(minibatch, nOut, 12345);
                    netSD.setInput(in);
                    netStandard.setInput(in);
                    netSD.setLabels(l);
                    netStandard.setLabels(l);

                    netSD.computeGradientAndScore();
                    netStandard.computeGradientAndScore();

                    Gradient gSD = netSD.gradient();
                    Gradient gStd = netStandard.gradient();

                    Map<String, INDArray> m1 = gSD.gradientForVariable();
                    Map<String, INDArray> m2 = gStd.gradientForVariable();

                    assertEquals(m2.keySet(), m1.keySet());

                    for (String s : m1.keySet()) {
                        INDArray i1 = m1.get(s);
                        INDArray i2 = m2.get(s);

                        assertEquals(s, i2, i1);
                    }

                    assertEquals(gStd.gradient(), gSD.gradient());
                }
            }
        }
    }

    @Test
    public void gradientCheck(){
        int nIn = 4;
        int nOut = 4;

        for(boolean workspaces : new boolean[]{false, true}) {
            for(Activation a : new Activation[]{Activation.TANH, Activation.IDENTITY}) {

                String msg = "workspaces: " + workspaces + ", " + a;
                Nd4j.getRandom().setSeed(12345);

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .updater(new NoOp())
                        .trainingWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                        .inferenceWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                        .list()
                        .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut).activation(a).build())
                        .layer(new SameDiffDense.Builder().nIn(nOut).nOut(nOut).activation(a).build())
                        .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        //.setInputType(InputType.feedForward(nIn))     //TODO
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                INDArray f = Nd4j.rand(3, nIn);
                INDArray l = TestUtils.randomOneHot(3, nOut);

                log.info("Starting: " + msg);
                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, f, l);

                assertTrue(msg, gradOK);
            }
        }
    }
}
