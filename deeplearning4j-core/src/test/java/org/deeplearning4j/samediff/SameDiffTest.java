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
                assertEquals(g2.gradient(), g.gradient());
            }
        }
    }

    @Test
    public void testShapeResolutionMinus1() {

        int nIn = 3;
        int nOut = 4;

        int minibatch = 3;

//        for(boolean useMinus1 : new boolean[]{false, true}) {
        for (boolean useMinus1 : new boolean[]{true}) {
            log.info("Starting: {}", (useMinus1 ? "minibatch -1" : "minibatch 3"));

            int[] inShape;
            if (useMinus1) {
                inShape = new int[]{-1, nIn};
            } else {
                inShape = new int[]{minibatch, nIn};
            }
            int[] wShape = new int[]{nIn, nOut};
            int[] bShape = new int[]{1, nOut};

            SameDiff sd = SameDiff.create();
            SDVariable layerInput = sd.var("in", inShape);
            SDVariable weights = sd.var("W", wShape);
            SDVariable bias = sd.var("b", bShape);

            SDVariable mmul = sd.mmul("mmul", layerInput, weights);
            SDVariable z = mmul.add("z", bias);
            SDVariable out = sd.sigmoid("out", z);

            INDArray in = Nd4j.rand(new int[]{minibatch, nIn});
            INDArray w = Nd4j.rand(wShape);
            INDArray b = Nd4j.rand(bShape);

            Map<String, INDArray> m = new HashMap<>();
            m.put("in", in);
            m.put("W", w);
            m.put("b", b);

            sd.associateArrayWithVariable(in, sd.getVariable("in"));
            sd.associateArrayWithVariable(w, sd.getVariable("W"));
            sd.associateArrayWithVariable(b, sd.getVariable("b"));

//            INDArray outArr = sd.execAndEndResult();

            sd.addAsPlaceHolder("in");
            sd.addAsPlaceHolder("W");
            sd.addAsPlaceHolder("b");

            sd.execWithPlaceHolder(m);

            INDArray outArr = sd.getVariable("out").getArr();

            assertArrayEquals(new int[]{minibatch, nOut}, outArr.shape());
        }
    }

    @Test
    public void debug() {

        int nIn = 3;
        int nOut = 4;

        int minibatch = 3;

        int[] inShape = new int[]{-1, nIn};
        int[] wShape = new int[]{nIn, nOut};
        int[] bShape = new int[]{1, nOut};

        SameDiff sd = SameDiff.create();
        SDVariable layerInput = sd.var("in", inShape);
        SDVariable weights = sd.var("W", wShape);
        SDVariable bias = sd.var("b", bShape);

        assertArrayEquals(inShape, layerInput.getShape());
        assertArrayEquals(wShape, weights.getShape());

        SDVariable mmul = sd.mmul("mmul", layerInput, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.sigmoid("out", z);

        INDArray in = Nd4j.rand(new int[]{minibatch, nIn});
        INDArray w = Nd4j.rand(wShape);
        INDArray b = Nd4j.rand(bShape);

        Map<String, INDArray> m = new HashMap<>();
        m.put("in", in);
        m.put("W", w);
        m.put("b", b);

        sd.associateArrayWithVariable(in, sd.getVariable("in"));
        sd.associateArrayWithVariable(w, sd.getVariable("W"));
        sd.associateArrayWithVariable(b, sd.getVariable("b"));

//            INDArray outArr = sd.execAndEndResult();

        sd.addAsPlaceHolder("in");
        sd.addAsPlaceHolder("W");
        sd.addAsPlaceHolder("b");

        sd.execWithPlaceHolder(m);

        INDArray outArr = sd.getVariable("out").getArr();

        assertArrayEquals(new int[]{minibatch, nOut}, outArr.shape());
    }

    @Test
    public void debug2() {
        int[] inShape = new int[]{-1, 3};

        SameDiff sd = SameDiff.create();
        SDVariable layerInput = sd.var("in", inShape);

        int[] actShape = layerInput.getShape(); //Getting: [1,3]
        assertArrayEquals(inShape, actShape);
    }

    @Test
    public void debugTransforms() {

        Activation[] afns = new Activation[]{
                //First 6 pass
                Activation.TANH, Activation.SIGMOID,
                Activation.ELU, Activation.IDENTITY, Activation.SOFTPLUS, Activation.SOFTSIGN,
                //Next 3 fail
                Activation.CUBE,    //Output differs
                Activation.HARDTANH,    //NPE
                Activation.RELU      //JVM crash
        };

        int nIn = 3;
        int nOut = 4;

        int minibatch = 3;

        int[] inShape = new int[]{minibatch, nIn};
        int[] wShape = new int[]{nIn, nOut};
        int[] bShape = new int[]{1, nOut};

        for (Activation a : afns) {
            log.info("Starting: " + a);
            SameDiff sd = SameDiff.create();
            SDVariable layerInput = sd.var("in", inShape);
            SDVariable weights = sd.var("W", wShape);
            SDVariable bias = sd.var("b", bShape);

            SDVariable mmul = sd.mmul("mmul", layerInput, weights);
            SDVariable z = mmul.add("z", bias);

            INDArray in = Nd4j.rand(new int[]{minibatch, nIn});
            INDArray w = Nd4j.rand(wShape);
            INDArray b = Nd4j.rand(bShape);

            INDArray exp = in.mmul(w).addiRowVector(b);

            SDVariable out = asSameDiff(a, "out", sd, z, exp);

            Map<String, INDArray> m = new HashMap<>();
            m.put("in", in);
            m.put("W", w);
            m.put("b", b);

            sd.associateArrayWithVariable(in, sd.getVariable("in"));
            sd.associateArrayWithVariable(w, sd.getVariable("W"));
            sd.associateArrayWithVariable(b, sd.getVariable("b"));

            sd.addAsPlaceHolder("in");
            sd.addAsPlaceHolder("W");
            sd.addAsPlaceHolder("b");

            sd.execWithPlaceHolder(m);

            INDArray outArr = sd.getVariable("out").getArr();

            assertEquals(exp, outArr);
        }
    }

    public static SDVariable asSameDiff(Activation a, String variableName, SameDiff sd, SDVariable input, INDArray input2) {
        switch (a) {
            case CUBE:
                Transforms.pow(input2, 3, false);
                return sd.pow(variableName, input, 3.0);
            case ELU:
                Transforms.elu(input2, false);
                return sd.elu(variableName, input);
            case HARDTANH:
                Transforms.hardTanh(input2, false);
                return sd.hardTanh(variableName, input);
            case IDENTITY:
                return input.add(variableName, 0.0);    //Hack to get new variable with same content
            case LEAKYRELU:
                Transforms.leakyRelu(input2, false);
                return sd.leakyRelu(variableName, input, 0.0);
            case RELU:
                Transforms.relu(input2, false);
                return sd.relu(variableName, input, 0.0);
            case SIGMOID:
                Transforms.sigmoid(input2, false);
                return sd.sigmoid(variableName, input);
            case SOFTMAX:
                Transforms.softmax(input2, false);
                return sd.softmax(variableName, input);
            case SOFTPLUS:
                Transforms.softPlus(input2, false);
                return sd.softplus(variableName, input);
            case SOFTSIGN:
                Transforms.softsign(input2, false);
                return sd.softsign(variableName, input);
            case TANH:
                Transforms.tanh(input2, false);
                return sd.tanh(variableName, input);
            case HARDSIGMOID:
            case RATIONALTANH:
            case RRELU:
            case RECTIFIEDTANH:
            case SELU:
            case SWISH:
            default:
                throw new UnsupportedOperationException("Activation function not yet supported: " + a);
        }
    }


    @Test
    public void debugMmul() {

        INDArray first = Nd4j.linspace(1, 3, 3);
        INDArray second = Nd4j.linspace(4, 7, 4);

        SameDiff sd = SameDiff.create();
        SDVariable f = sd.var("in1", first);
        SDVariable s = sd.var("in2", second);
        SDVariable fTranspose = sd.transpose(f);
        SDVariable mmul = sd.mmul("mmul", fTranspose, s);

        INDArray out = sd.execAndEndResult();

        INDArray exp = first.transpose().mmul(second);
        assertEquals(exp, out);
    }

    @Test
    public void debugMmul2() {
        //Here: [1,3]^T * [1,4] = [3,4]

        INDArray first = Nd4j.linspace(1, 3, 3);
        INDArray second = Nd4j.linspace(4, 7, 4);

        SameDiff sd = SameDiff.create();
        SDVariable f = sd.var("in1", first);
        SDVariable s = sd.var("in2", second);

        MMulTranspose mt = MMulTranspose.builder()
                .transposeA(true)
                .transposeB(false)
                .transposeResult(false)
                .a(first)
                .b(second)
                .build();
        SDVariable mmul = sd.f().mmul(f, s, mt);
        sd.updateVariableNameAndReference(mmul, "mmul");

        INDArray out = sd.execAndEndResult();

        INDArray exp = first.transpose().mmul(second);
        assertEquals(exp, out);
    }
}
