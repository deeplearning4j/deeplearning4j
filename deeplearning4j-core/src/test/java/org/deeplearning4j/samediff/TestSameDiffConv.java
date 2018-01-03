package org.deeplearning4j.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
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
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

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

        int imgH = 3;
        int imgW = 3;
        int count = 0;
        int minibatch = 1;
        boolean hasBias = false;
        int nIn = 1;
        int nOut = 1;
        int[] kernel = {2, 2};
        int[] strides = {1, 1};
        int[] dilation = {2, 1};
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
//                        .dilation(new int[]{dilation[1], dilation[0]})
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

        int imgH = 16;
        int imgW = 20;

        int count = 0;

        //Note: to avoid the exporential number of tests here, we'll randomly run every Nth test only.
        //With n=1, m=3 this is 1 out of every 3 tests (on average)
        Random r = new Random(12345);
        int n = 1;
        int m = 3;
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
                                            int i = r.nextInt(m);
                                            if (i >= n) {
                                                //Example: n=2, m=3... skip on i=2, run test on i=0, i=1
                                                continue;
                                            }

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
                                                    .layer(new SameDiffConv.Builder()
                                                            .nIn(nOut)
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
                                                    .layer(new ConvolutionLayer.Builder()
                                                            .nIn(nOut)
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
    public void testConv2dEdgeCase(){
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);

        INDArray in = Nd4j.create(1,1,1,4).assign(Nd4j.linspace(1,4,4)).muli(10);      //NCHW
        INDArray wArr = Nd4j.create(4,1,1,1);       //dOut, dIt, kH, kW
        wArr.get(all(), point(0), point(0), point(0)).assign(Nd4j.linspace(1,4,4)).addi(0.5);

        SameDiff sd = SameDiff.create();
        SDVariable i = sd.var("in", in);
        SDVariable w = sd.var("w", wArr);

        Conv2DConfig conf = Conv2DConfig.builder()
                .isSameMode(false)
                .kw(1)
                .kh(1)
                .dh(1)
                .dw(1)
                .sy(1)
                .sx(1)
                .ph(0)
                .pw(0)
                .build();

        SDVariable conv2d = sd.conv2d(new SDVariable[]{i,w}, conf);

        INDArray out = sd.execAndEndResult();


        //1x1 conv edge case: equivalent to linear op for each position. Also: depth 1 in allows us to use concat + mul here
        INDArray wVec = wArr.get( all(), point(0), point(0), point(0));
        INDArray exp = Nd4j.concat(1, in, in, in, in);
        Nd4j.getExecutioner().exec(new BroadcastMulOp(exp, wVec, exp, 1));

        for(int j=0; j<4; j++ ){
            System.out.println(exp.get(point(0), point(j), all(), all()) + "\t" + out.get(point(0), point(j), all(), all()));
        }

        assertEquals(exp, out);
    }


    @Test
    public void testConv2dEdgeCase2(){

        INDArray in = Nd4j.create(1,1,1,4).assign(Nd4j.linspace(1,4,4)).muli(10);      //NCHW
        INDArray wArr = Nd4j.create(3,1,1,1);       //dOut, dIt, kH, kW
        wArr.get(all(), point(0), point(0), point(0)).assign(Nd4j.linspace(1,3,3)).addi(0.5);

        SameDiff sd = SameDiff.create();
        SDVariable i = sd.var("in", in);
        SDVariable w = sd.var("w", wArr);

        Conv2DConfig conf = Conv2DConfig.builder()
                .isSameMode(false)
                .kw(1)
                .kh(1)
                .dh(1)
                .dw(1)
                .sy(1)
                .sx(1)
                .ph(0)
                .pw(0)
                .build();

        SDVariable conv2d = sd.conv2d(new SDVariable[]{i,w}, conf);

        INDArray out = sd.execAndEndResult();


        //1x1 conv edge case: equivalent to linear op for each position. Also: depth 1 in allows us to use concat + mul here
        INDArray wVec = wArr.get( all(), point(0), point(0), point(0));
        INDArray exp = Nd4j.concat(1, in, in, in);
        Nd4j.getExecutioner().exec(new BroadcastMulOp(exp, wVec, exp, 1));

        for(int j=0; j<3; j++ ){
            System.out.println(exp.get(point(0), point(j), all(), all()) + "\t" + out.get(point(0), point(j), all(), all()));
        }

        assertEquals(exp, out);
    }

    @Test
    public void testConv2dEdgeCase3(){

        INDArray in = Nd4j.create(1,1,1,3).assign(Nd4j.linspace(1,3,3)).muli(10);      //NCHW
        INDArray wArr = Nd4j.create(4,1,1,1);       //dOut, dIt, kH, kW
        wArr.get(all(), point(0), point(0), point(0)).assign(Nd4j.linspace(1,4,4)).addi(0.5);

        SameDiff sd = SameDiff.create();
        SDVariable i = sd.var("in", in);
        SDVariable w = sd.var("w", wArr);

        Conv2DConfig conf = Conv2DConfig.builder()
                .isSameMode(false)
                .kw(1)
                .kh(1)
                .dh(1)
                .dw(1)
                .sy(1)
                .sx(1)
                .ph(0)
                .pw(0)
                .build();

        SDVariable conv2d = sd.conv2d(new SDVariable[]{i,w}, conf);

        INDArray out = sd.execAndEndResult();


        //1x1 conv edge case: equivalent to linear op for each position. Also: depth 1 in allows us to use concat + mul here
        INDArray wVec = wArr.get( all(), point(0), point(0), point(0));
        INDArray exp = Nd4j.concat(1, in, in, in, in);
        Nd4j.getExecutioner().exec(new BroadcastMulOp(exp, wVec, exp, 1));

        for(int j=0; j<4; j++ ){
            System.out.println(exp.get(point(0), point(j), all(), all()) + "\t" + out.get(point(0), point(j), all(), all()));
        }

        assertEquals(exp, out);
    }
}
