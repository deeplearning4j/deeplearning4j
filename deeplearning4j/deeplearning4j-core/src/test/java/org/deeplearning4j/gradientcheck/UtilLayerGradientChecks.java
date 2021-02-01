/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.util.MaskLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertTrue;

public class UtilLayerGradientChecks extends BaseDL4JTest {

    static {
        Nd4j.setDataType(DataType.DOUBLE);
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    @Test
    public void testMaskLayer() {
        Nd4j.getRandom().setSeed(12345);
        int tsLength = 3;

        for(int minibatch : new int[]{1,3}) {
            for (int inputRank : new int[]{2, 3, 4}) {
                for (boolean inputMask : new boolean[]{false, true}) {
                    String maskType = (inputMask ? "inputMask" : "none");

                    INDArray inMask = null;
                    if (inputMask) {
                        switch (inputRank) {
                            case 2:
                                if(minibatch == 1){
                                    inMask = Nd4j.ones(1,1);
                                } else {
                                    inMask = Nd4j.create(DataType.DOUBLE, minibatch, 1);
                                    Nd4j.getExecutioner().exec(new BernoulliDistribution(inMask, 0.5));
                                    int count = inMask.sumNumber().intValue();
                                    assertTrue(count >= 0 && count <= minibatch);   //Sanity check on RNG seed
                                }
                                break;
                            case 4:
                                //Per-example mask (broadcast along all channels/x/y)
                                if(minibatch == 1){
                                    inMask = Nd4j.ones(DataType.DOUBLE, 1,1, 1, 1);
                                } else {
                                    inMask = Nd4j.create(DataType.DOUBLE, minibatch, 1, 1, 1);
                                    Nd4j.getExecutioner().exec(new BernoulliDistribution(inMask, 0.5));
                                    int count = inMask.sumNumber().intValue();
                                    assertTrue(count >= 0 && count <= minibatch);   //Sanity check on RNG seed
                                }
                                break;
                            case 3:
                                inMask = Nd4j.ones(DataType.DOUBLE, minibatch, tsLength);
                                for( int i=0; i<minibatch; i++ ){
                                    for( int j=i+1; j<tsLength; j++ ){
                                        inMask.putScalar(i,j,0.0);
                                    }
                                }
                                break;
                            default:
                                throw new RuntimeException();
                        }
                    }

                    int[] inShape;
                    int[] labelShape;
                    switch (inputRank){
                        case 2:
                            inShape = new int[]{minibatch, 3};
                            labelShape = inShape;
                            break;
                        case 3:
                            inShape = new int[]{minibatch, 3, tsLength};
                            labelShape = inShape;
                            break;
                        case 4:
                            inShape = new int[]{minibatch, 1, 5, 5};
                            labelShape = new int[]{minibatch, 5};
                            break;
                        default:
                            throw new RuntimeException();
                    }
                    INDArray input = Nd4j.rand(inShape).muli(100);
                    INDArray label = Nd4j.rand(labelShape);

                    String name = "mb=" + minibatch + ", maskType=" + maskType + ", inputRank=" + inputRank;
                    System.out.println("*** Starting test: " + name);

                    Layer l1;
                    Layer l2;
                    Layer l3;
                    InputType it;
                    switch (inputRank){
                        case 2:
                            l1 = new DenseLayer.Builder().nOut(3).build();
                            l2 = new DenseLayer.Builder().nOut(3).build();
                            l3 = new OutputLayer.Builder().nOut(3).lossFunction(LossFunctions.LossFunction.MSE)
                                    .activation(Activation.TANH).build();
                            it = InputType.feedForward(3);
                            break;
                        case 3:
                            l1 = new SimpleRnn.Builder().nIn(3).nOut(3).activation(Activation.TANH).build();
                            l2 = new SimpleRnn.Builder().nIn(3).nOut(3).activation(Activation.TANH).build();
                            l3 = new RnnOutputLayer.Builder().nIn(3).nOut(3).lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                                    .activation(Activation.IDENTITY).build();
                            it = InputType.recurrent(3);
                            break;
                        case 4:
                            l1 = new ConvolutionLayer.Builder().nOut(5).convolutionMode(ConvolutionMode.Truncate)
                                    .stride(1,1).kernelSize(2,2).padding(0,0)
                                    .build();
                            l2 = new ConvolutionLayer.Builder().nOut(5).convolutionMode(ConvolutionMode.Truncate)
                                    .stride(1,1).kernelSize(2,2).padding(0,0)
                                    .build();
                            l3 = new OutputLayer.Builder().nOut(5).lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                                    .activation(Activation.IDENTITY)
                                    .build();
                            it = InputType.convolutional(5,5,1);
                            break;
                        default:
                            throw new RuntimeException();

                    }

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .updater(new NoOp())
                            .activation(Activation.TANH)
                            .dataType(DataType.DOUBLE)
                            .dist(new NormalDistribution(0,2))
                            .list()
                            .layer(l1)
                            .layer(new MaskLayer())
                            .layer(l2)
                            .layer(l3)
                            .setInputType(it)
                            .build();


                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(net)
                            .minAbsoluteError(1e-6)
                            .input(input).labels(label).inputMask(inMask));
                    assertTrue(gradOK);

                    TestUtils.testModelSerialization(net);
                }
            }
        }
    }


    @Test
    public void testFrozenWithBackprop(){

        for( int minibatch : new int[]{1,5}) {

            MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .dataType(DataType.DOUBLE)
                    .seed(12345)
                    .updater(Updater.NONE)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(10).nOut(10)
                            .activation(Activation.TANH).weightInit(WeightInit.XAVIER).build())
                    .layer(new FrozenLayerWithBackprop(new DenseLayer.Builder().nIn(10).nOut(10)
                            .activation(Activation.TANH).weightInit(WeightInit.XAVIER).build()))
                    .layer(new FrozenLayerWithBackprop(
                            new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.TANH)
                                    .weightInit(WeightInit.XAVIER).build()))
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX).nIn(10).nOut(10).build())
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf2);
            net.init();

            INDArray in = Nd4j.rand(minibatch, 10);
            INDArray labels = TestUtils.randomOneHot(minibatch, 10);

            Set<String> excludeParams = new HashSet<>();
            excludeParams.addAll(Arrays.asList("1_W", "1_b", "2_W", "2_b"));

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(net).input(in)
                    .labels(labels).excludeParams(excludeParams));
            assertTrue(gradOK);

            TestUtils.testModelSerialization(net);


            //Test ComputationGraph equivalent:
            ComputationGraph g = net.toComputationGraph();

            boolean gradOKCG = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(g)
                    .minAbsoluteError(1e-6)
                    .inputs(new INDArray[]{in}).labels(new INDArray[]{labels}).excludeParams(excludeParams));
            assertTrue(gradOKCG);

            TestUtils.testModelSerialization(g);
        }

    }
}
