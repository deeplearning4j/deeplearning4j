package org.deeplearning4j.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.samediff.testlayers.SameDiffConv;
import org.deeplearning4j.samediff.testlayers.SameDiffDense;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.*;

@Slf4j
public class TestSameDiffConv {

    @Test
    public void testSameDiffConvBasic() {

        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new SameDiffConv.Builder().nIn(nIn).nOut(nOut).kernelSize(kH, kW).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Map<String, INDArray> pt1 = net.getLayer(0).paramTable();
        assertNotNull(pt1);
        assertEquals(2, pt1.size());
        assertNotNull(pt1.get(ConvolutionParamInitializer.WEIGHT_KEY));
        assertNotNull(pt1.get(ConvolutionParamInitializer.BIAS_KEY));

        assertArrayEquals(new int[]{nOut, nIn, kH, kW}, pt1.get(ConvolutionParamInitializer.WEIGHT_KEY).shape());
        assertArrayEquals(new int[]{1, nOut}, pt1.get(ConvolutionParamInitializer.BIAS_KEY).shape());

        TestUtils.testModelSerialization(net);
    }

    @Test
    public void testSameDiffConvForward_Debug() {

        int imgH = 16;
        int imgW = 24;
        int count = 0;
        int minibatch = 5;
        boolean hasBias = true;
        int nIn = 3;
        int nOut = 4;
        int[] kernel = {2, 2};
        int[] strides = {1, 1};
        int[] dilation = {1, 1};
        ConvolutionMode cm = ConvolutionMode.Truncate;
        Activation a = Activation.TANH;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new SameDiffConv.Builder()
                        .nIn(nIn)
                        .nOut(nOut)
                        .kernelSize(kernel)
                        .stride(strides)
                        .dilation(dilation)
                        .convolutionMode(cm)
                        .activation(a)
                        .hasBias(hasBias)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertNotNull(net.paramTable());

        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .nIn(nIn)
                        .nOut(nOut)
                        .kernelSize(kernel)
                        .stride(strides)
                        .dilation(dilation)
                        .convolutionMode(cm)
                        .activation(a)
                        .hasBias(hasBias)
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

        INDArray in = Nd4j.rand(new int[]{minibatch, nIn, imgH, imgW});
        INDArray out = net.output(in);
        INDArray outExp = net2.output(in);

        assertEquals(outExp, out);

        //Also check serialization:
        MultiLayerNetwork netLoaded = TestUtils.testModelSerialization(net);
        INDArray outLoaded = netLoaded.output(in);

        assertEquals(outExp, outLoaded);
    }

    @Test
    public void testSameDiffConvForward() {

        int imgH = 8;
        int imgW = 12;

        int count = 0;
        for (int minibatch : new int[]{5, 1}) {

            Activation[] afns = new Activation[]{
                    Activation.TANH,
                    Activation.SIGMOID,
                    Activation.ELU,
                    Activation.IDENTITY,
                    Activation.SOFTPLUS,
                    Activation.SOFTSIGN,
//                    Activation.CUBE,    //https://github.com/deeplearning4j/nd4j/issues/2426
                    Activation.HARDTANH,
                    Activation.RELU
            };

            for (boolean hasBias : new boolean[]{true, false}) {
                for (int nIn : new int[]{3, 4}) {
                    for (int nOut : new int[]{4, 5}) {
                        for (int[] kernel : new int[][]{{2, 2}, {2, 1}, {3, 2}}) {
                            for (int[] strides : new int[][]{{1, 1}, {2, 2}, {2, 1}}) {
                                for (int[] dilation : new int[][]{{1, 1}, {2, 2}, {1, 2}}) {
                                    for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Truncate, ConvolutionMode.Same}) {
                                        for (Activation a : afns) {
                                            String msg = "Test " + (count++) + " - minibatch=" + minibatch + ", nIn=" + nIn
                                                    + ", nOut=" + nOut + ", kernel=" + Arrays.toString(kernel) + ", stride="
                                                    + Arrays.toString(strides) + ", dilation=" + Arrays.toString(dilation)
                                                    + ", ConvolutionMode=" + cm + ", ActFn=" + a + ", hasBias=" + hasBias;
                                            log.info("Starting test: " + msg);

                                            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                                    .list()
                                                    .layer(new SameDiffConv.Builder()
                                                            .nIn(nIn)
                                                            .nOut(nOut)
                                                            .kernelSize(kernel)
                                                            .stride(strides)
                                                            .dilation(dilation)
                                                            .convolutionMode(cm)
                                                            .activation(a)
                                                            .hasBias(hasBias)
                                                            .build())
                                                    .build();

                                            MultiLayerNetwork net = new MultiLayerNetwork(conf);
                                            net.init();

                                            assertNotNull(net.paramTable());

                                            MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                                                    .list()
                                                    .layer(new ConvolutionLayer.Builder()
                                                            .nIn(nIn)
                                                            .nOut(nOut)
                                                            .kernelSize(kernel)
                                                            .stride(strides)
                                                            .dilation(dilation)
                                                            .convolutionMode(cm)
                                                            .activation(a)
                                                            .hasBias(hasBias)
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

                                            INDArray in = Nd4j.rand(new int[]{minibatch, nIn, imgH, imgW});
                                            INDArray out = net.output(in);
                                            INDArray outExp = net2.output(in);

                                            assertEquals(msg, outExp, out);

                                            //Also check serialization:
                                            MultiLayerNetwork netLoaded = TestUtils.testModelSerialization(net);
                                            INDArray outLoaded = netLoaded.output(in);

                                            assertEquals(msg, outExp, outLoaded);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testConv2dBasic() {
        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.create(nOut, nIn, kH, kW); //As per DL4J
        INDArray bArr = Nd4j.create(1, nOut);
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);
        SDVariable b = sd.var("b", bArr);

        //Order: https://github.com/deeplearning4j/libnd4j/blob/6c41ea5528bb1f454e92a9da971de87b93ff521f/include/ops/declarable/generic/convo/conv2d.cpp#L20-L22
        //in, w, b - bias is optional
        SDVariable[] vars = new SDVariable[]{in, w, b};

        Conv2DConfig c = Conv2DConfig.builder()
                .kh(kH).kw(kW)
                .ph(0).pw(0)
                .sy(1).sx(1)
                .dh(1).dw(1)
                .isSameMode(false)
                .build();

        SDVariable conv = sd.conv2d(vars, c);

        SDVariable out = sd.tanh("tanh", conv);
//        SDVariable out = conv.add("out", 1.0);
//        SDVariable out = sd.sum(conv, 1,2,3);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        int[] outShape = outArr.shape();
        assertArrayEquals(new int[]{mb, nOut, 27, 27}, outShape);
    }
}
