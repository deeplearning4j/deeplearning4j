/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.cuda.convolution;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.cuda.CuDNNTestUtils;
import org.deeplearning4j.cuda.TestUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.resources.Resources;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Created by Alex on 15/11/2016.
 */
public class TestConvolution extends BaseDL4JTest {


    @Override
    public long getTimeoutMilliseconds() {
        return 240000L;
    }

    @Test
    public void testSameModeActivationSizes() {
        int inH = 3;
        int inW = 4;
        int inDepth = 3;
        int minibatch = 5;

        int sH = 2;
        int sW = 2;
        int kH = 3;
        int kW = 3;

        org.deeplearning4j.nn.conf.layers.Layer[] l = new org.deeplearning4j.nn.conf.layers.Layer[2];
        l[0] = new ConvolutionLayer.Builder().nOut(4).kernelSize(kH, kW).stride(sH, sW).build();
        l[1] = new SubsamplingLayer.Builder().kernelSize(kH, kW).stride(sH, sW).build();

        for (int i = 0; i < l.length; i++) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().convolutionMode(ConvolutionMode.Same)
                    .list().layer(0, l[i]).layer(1, new OutputLayer.Builder().nOut(3).build())
                    .setInputType(InputType.convolutional(inH, inW, inDepth)).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            INDArray inData = Nd4j.create(minibatch, inDepth, inH, inW);
            List<INDArray> activations = net.feedForward(inData);
            INDArray actL0 = activations.get(1);

            int outH = (int) Math.ceil(inH / ((double) sH));
            int outW = (int) Math.ceil(inW / ((double) sW));

            System.out.println(Arrays.toString(actL0.shape()));
            assertArrayEquals(new long[]{minibatch, (i == 0 ? 4 : inDepth), outH, outW}, actL0.shape());
        }
    }


    @Test
    public void testCompareCudnnStandardOutputsVsMode() throws Exception {

        ConvolutionMode[] cm =
                new ConvolutionMode[]{ConvolutionMode.Strict, ConvolutionMode.Truncate, ConvolutionMode.Same};

        for (ConvolutionMode c : cm) {
            for (ConvolutionLayer.AlgoMode a : new ConvolutionLayer.AlgoMode[]{ConvolutionLayer.AlgoMode.NO_WORKSPACE, ConvolutionLayer.AlgoMode.PREFER_FASTEST}) {
                for (boolean conv : new boolean[]{true, false}) {
                    String msg = c + " - " + a + " - " + (conv ? "conv" : "subsampling");
                    System.out.println(msg);

                    org.deeplearning4j.nn.conf.layers.Layer l;
                    if (conv) {
                        l = new ConvolutionLayer.Builder().nOut(4).kernelSize(4, 4).stride(2, 2).build();
                    } else {
                        l = new SubsamplingLayer.Builder().kernelSize(4, 4).stride(2, 2).build();
                    }

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .dataType(DataType.DOUBLE)
                            .seed(12345)
                            .l2(0.0005).updater(new Sgd(0.01)).weightInit(WeightInit.XAVIER).convolutionMode(c).cudnnAlgoMode(a).list()
                            .layer(0, l)
                            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                    .nOut(10).activation(Activation.SOFTMAX).build())
                            .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
                            .build();
                    if (conv) {
                        assertEquals(a, ((ConvolutionLayer) l).getCudnnAlgoMode());
                    }

                    Nd4j.getRandom().setSeed(12345);
                    MultiLayerNetwork net1 = new MultiLayerNetwork(conf);
                    net1.init();
                    net1.initGradientsView();

                    Nd4j.getRandom().setSeed(12345);
                    MultiLayerNetwork net2 = new MultiLayerNetwork(conf);
                    net2.init();
                    net2.initGradientsView();

                    Layer layerCudnn = net1.getLayer(0);
                    Layer layerStandard = net2.getLayer(0);

                    Field f = layerStandard.getClass().getDeclaredField("helper");
                    f.setAccessible(true);
                    f.set(layerStandard, null);

                    if (f.get(layerCudnn) == null)
                        throw new RuntimeException();
                    if (f.get(layerStandard) != null)
                        throw new RuntimeException();


                    INDArray in = Nd4j.rand(DataType.DOUBLE, new int[]{1, 1, 20, 20}); //(20-4+0)/2 +1 = 9

                    INDArray outCudnn = layerCudnn.activate(in, false, LayerWorkspaceMgr.noWorkspaces());
                    INDArray outStd = layerStandard.activate(in, false, LayerWorkspaceMgr.noWorkspaces());

                    assertEquals(outStd, outCudnn, msg);


                    //Check backprop:
                    INDArray epsilon = Nd4j.rand(DataType.DOUBLE, outStd.shape());
                    Pair<Gradient, INDArray> pCudnn = layerCudnn.backpropGradient(epsilon.dup(), LayerWorkspaceMgr.noWorkspaces());
                    Pair<Gradient, INDArray> pStd = layerStandard.backpropGradient(epsilon.dup(), LayerWorkspaceMgr.noWorkspaces());

//                    System.out.println(Arrays.toString(pStd.getSecond().data().asFloat()));
//                    System.out.println(Arrays.toString(pCudnn.getSecond().data().asFloat()));

                    INDArray epsOutStd = pStd.getSecond();
                    INDArray epsOutCudnn = pCudnn.getSecond();

                    assertTrue(epsOutStd.equalsWithEps(epsOutCudnn, 1e-4), msg);

                    if (conv) {
                        INDArray gradStd = pStd.getFirst().gradient();
                        INDArray gradCudnn = pCudnn.getFirst().gradient();

                        assertTrue(gradStd.equalsWithEps(gradCudnn, 1e-4), msg);
                    }
                }
            }
        }
    }


    @Test
    public void validateXceptionImport(@TempDir Path testDir) throws Exception {
        File dir = testDir.toFile();
        File fSource = Resources.asFile("modelimport/keras/examples/xception/xception_tf_keras_2.h5");
        File fExtracted = new File(dir, "xception_tf_keras_2.h5" );
        FileUtils.copyFile(fSource, fExtracted);

        int inSize = 256;
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights( fExtracted.getAbsolutePath(), new int[]{inSize, inSize, 3}, false);
        model = model.convertDataType(DataType.DOUBLE);

        INDArray in = Nd4j.rand(DataType.DOUBLE, new int[]{1, inSize, inSize, 3});      //Keras import model -> NHWC

        CuDNNTestUtils.assertHelpersPresent(model.getLayers());
        Map<String,INDArray> withCudnn = model.feedForward(in, false);

        CuDNNTestUtils.removeHelpers(model.getLayers());
        CuDNNTestUtils.assertHelpersAbsent(model.getLayers());
        Map<String,INDArray> noCudnn = model.feedForward(in, false);

        assertEquals(withCudnn.keySet(), noCudnn.keySet());

        for(String s : withCudnn.keySet()) {
            assertEquals(withCudnn.get(s), noCudnn.get(s), s);
        }
    }


    @Test
    public void testCudnnDilation(){
        //Sanity check on dilated conv execution
        int[] k = new int[]{2,3,4,5};
        int[] d = new int[]{1,2,3,4};

        for( int[] inputSize : new int[][]{{10,1,28,28}, {3,3,224,224}}) {
            for (int i = 0; i < k.length; i++) {
                for(ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Same, ConvolutionMode.Truncate}) {

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .convolutionMode(ConvolutionMode.Same)
                            .list()
                            .layer(new ConvolutionLayer.Builder().kernelSize(k[i], k[i]).dilation(d[i], d[i]).nOut(3).build())
                            .layer(new SubsamplingLayer.Builder().kernelSize(k[i], k[i]).dilation(d[i], d[i]).build())
                            .layer(new OutputLayer.Builder().nOut(10).build())
                            .setInputType(InputType.convolutional(inputSize[3], inputSize[2], inputSize[1]))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    INDArray in = Nd4j.create(inputSize);
                    net.output(in);
                }
            }
        }
    }


    @Test
    public void testGradientNorm() throws Exception {

        int height = 100;
        int width = 100;
        int channels = 1;
        int numLabels = 10;

        for( int batchSize : new int[]{1, 32}) {

            long seed = 12345;
            double nonZeroBias = 1;

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .dataType(DataType.DOUBLE)
                    .dist(new NormalDistribution(0.0, 0.01))
                    .activation(Activation.RELU)
                    .updater(new Adam(5e-3))
                    //.biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 20000), 0.9))
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .l2(5 * 1e-4)
                    .list()
                    .layer(convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4},
                            new int[]{3, 3}, 0))
                    .layer(maxPool("maxpool1", new int[]{3, 3}))
                    .layer(conv5x5("cnn2", 256, new int[]{1, 1}, new int[]{2, 2}, nonZeroBias))
                    .layer(maxPool("maxpool2", new int[]{3, 3}))
                    .layer(conv3x3("cnn3", 384, 0))
                    .layer(conv3x3("cnn4", 384, nonZeroBias))
                    .layer(conv3x3("cnn5", 256, nonZeroBias))
                    .layer(maxPool("maxpool3", new int[]{3, 3}))
                    .layer(fullyConnected("ffn1", 4096, nonZeroBias, new GaussianDistribution(0, 0.005)))
                    .layer(fullyConnected("ffn2", 4096, nonZeroBias, new GaussianDistribution(0, 0.005)))
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .name("output")
                            .nOut(numLabels)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .setInputType(InputType.convolutional(height, width, channels))
                    .build();


            MultiLayerNetwork netNoCudnn = new MultiLayerNetwork(conf.clone());
            netNoCudnn.init();
            MultiLayerNetwork netWithCudnn = new MultiLayerNetwork(conf.clone());
            netWithCudnn.init();

            CuDNNTestUtils.removeHelpers(netNoCudnn.getLayers());



            Nd4j.getRandom().setSeed(12345);
            for( int j=0; j<3; j++ ) {
//                System.out.println("j=" + j);
                INDArray f = Nd4j.rand(new int[]{batchSize, channels, height, width});
                INDArray l = TestUtils.randomOneHot(batchSize, numLabels);

                netNoCudnn.fit(f, l);
                netWithCudnn.fit(f, l);

                assertEquals(netNoCudnn.score(), netWithCudnn.score(), 1e-5);

                for (Map.Entry<String, INDArray> e : netNoCudnn.paramTable().entrySet()) {
                    boolean pEq = e.getValue().equalsWithEps(netWithCudnn.paramTable().get(e.getKey()), 1e-4);
//                    int idx = e.getKey().indexOf("_");
//                    int layerNum = Integer.parseInt(e.getKey().substring(0, idx));
                    //System.out.println(e.getKey() + " - " + pEq + " - " + netNoCudnn.getLayer(layerNum).getClass().getSimpleName());
                    assertTrue(pEq);
                }

                boolean eq = netNoCudnn.params().equalsWithEps(netWithCudnn.params(), 1e-4);
                assertTrue(eq);
            }
        }
    }


    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride,
                                             int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name)
                .nIn(in)
                .nOut(out)
                .biasInit(bias)
                .build();
    }

    private static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[] { 3, 3 }, new int[] { 1, 1 },
                new int[] { 1, 1 }).name(name).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad,
                                            double bias) {
        return new ConvolutionLayer.Builder(new int[] { 5, 5 }, stride, pad).name(name)
                .nOut(out)
                .biasInit(bias)
                .build();
    }

    private static SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[] { 2, 2 }).name(name).build();
    }

    private static DenseLayer fullyConnected(String name, int out, double bias, Distribution dist) {
        return new DenseLayer.Builder().name(name)
                .nOut(out)
                .biasInit(bias)
                .dist(dist)
                .build();
    }
}
