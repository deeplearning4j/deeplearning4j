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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.*;

@Slf4j
public class SameDiffTestConv {

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
    }

    @Test
    public void testSameDiffConvForward() {

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
                    Activation.HARDTANH,    //NPE
//                 Activation.RELU      //JVM crash
            };

            for(int nIn : new int[]{3,4}){
                for( int nOut : new int[]{4,5}){
                    for( int[] kernel : new int[][]{{2,2}, {2,1}, {3,2}}){
                        for( int[] strides : new int[][]{{1,1}, {2,2}, {2,1}}){
                            for( int[] dilation : new int[][]{{1,1}, {2,2}, {1,2}}){
                                for(ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Truncate, ConvolutionMode.Same}){
                                    for(Activation a : afns){
                                        String msg = "Test " + (count++) + " - minibatch=" + minibatch + ", nIn=" + nIn
                                                + ", nOut=" + nOut + ", kernel=" + Arrays.toString(kernel) + ", stride="
                                                + Arrays.toString(strides) + ", dilation=" + Arrays.toString(dilation)
                                                + ", ConvolutionMode=" + cm + ", ActFn=" + a;
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
