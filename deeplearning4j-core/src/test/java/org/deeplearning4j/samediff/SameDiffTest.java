package org.deeplearning4j.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.samediff.testlayers.SameDiffDense;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

@Slf4j
public class SameDiffTest {

    @Test
    public void testSameDiffDenseBasic() {

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

        assertArrayEquals(new int[]{nIn, nOut}, pt1.get(DefaultParamInitializer.WEIGHT_KEY).shape());
        assertArrayEquals(new int[]{1, nOut}, pt1.get(DefaultParamInitializer.BIAS_KEY).shape());
    }

    @Test
    public void testSameDiffDenseForward() {

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
//                    Activation.CUBE,    //https://github.com/deeplearning4j/nd4j/issues/2426
                    Activation.HARDTANH,    //NPE
//                 Activation.RELU      //JVM crash
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
            }
        }
    }

    @Test
    public void testSameDiffDenseBackward() {

        int nIn = 3;
        int nOut = 4;

        for (int minibatch : new int[]{5, 1}) {

            Activation[] afns = new Activation[]{
                    Activation.TANH,
                    Activation.SIGMOID,
                    Activation.ELU, Activation.IDENTITY, Activation.SOFTPLUS, Activation.SOFTSIGN,
                    Activation.HARDTANH,
//                    Activation.CUBE,    //https://github.com/deeplearning4j/nd4j/issues/2426
//                    Activation.RELU      //JVM crash
            };

            for (Activation a : afns) {
                log.info("Starting test - " + a);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut)
                                .activation(a)
                                .build())
                        .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(new DenseLayer.Builder().activation(a).nIn(nIn).nOut(nOut).build())
                        .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .build();

                MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
                net2.init();

                net.params().assign(net2.params());

                //Check params:
                assertEquals(net2.params(), net.params());
                assertEquals(net2.paramTable(), net.paramTable());

                INDArray in = Nd4j.rand(minibatch, nIn);
                INDArray l = TestUtils.randomOneHot(minibatch, nOut, 12345);
                net.setInput(in);
                net2.setInput(in);
                net.setLabels(l);
                net2.setLabels(l);

                net.computeGradientAndScore();
                net2.computeGradientAndScore();

                Gradient g = net.gradient();
                Gradient g2 = net2.gradient();

                Map<String,INDArray> m1 = g.gradientForVariable();
                Map<String,INDArray> m2 = g2.gradientForVariable();

                assertEquals(m2.keySet(), m1.keySet());

                for(String s : m1.keySet()){
                    INDArray i1 = m1.get(s);
                    INDArray i2 = m2.get(s);

                    assertEquals(s, i2, i1);
                }

                assertEquals(g2.gradient(), g.gradient());
            }
        }
    }
}
