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
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.util.MaskLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

import static org.junit.Assert.assertTrue;

public class UtilLayerGradientChecks extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-6;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testMaskLayer() {
        Nd4j.getRandom().setSeed(12345);
        int tsLength = 5;

        for(int minibatch : new int[]{1,8}) {
            for (int inputRank : new int[]{2, 3, 4}) {
                for (boolean inputMask : new boolean[]{false, true}) {
                    String maskType = (inputMask ? "inputMask" : "none");

                    INDArray inMask = null;
                    if (inputMask) {
                        switch (inputRank) {
                            case 2:
                            case 4:
                                if(minibatch == 1){
                                    inMask = Nd4j.ones(1,1);
                                } else {
                                    inMask = Nd4j.create(minibatch, 1);
                                    Nd4j.getExecutioner().exec(new BernoulliDistribution(inMask, 0.5));
                                    int count = inMask.sumNumber().intValue();
                                    assertTrue(count >= 0 && count <= minibatch);   //Sanity check on RNG seed
                                }
                                break;
                            case 3:
                                inMask = Nd4j.ones(minibatch, tsLength);
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
                            inShape = new int[]{minibatch, 5};
                            labelShape = inShape;
                            break;
                        case 3:
                            inShape = new int[]{minibatch, 5, tsLength};
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
                            l1 = new DenseLayer.Builder().nOut(5).build();
                            l2 = new DenseLayer.Builder().nOut(5).build();
                            l3 = new OutputLayer.Builder().nOut(5).lossFunction(LossFunctions.LossFunction.MSE)
                                    .activation(Activation.TANH).build();
                            it = InputType.feedForward(5);
                            break;
                        case 3:
                            l1 = new LSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build();
                            l2 = new LSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build();
                            l3 = new RnnOutputLayer.Builder().nIn(5).nOut(5).lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                                    .activation(Activation.IDENTITY).build();
                            it = InputType.recurrent(5);
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
                            .weightInit(WeightInit.DISTRIBUTION)
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


                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, label, inMask, null);
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

            boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, null, null,
                    false, -1, excludeParams);
            assertTrue(gradOK);

            TestUtils.testModelSerialization(net);


            //Test ComputationGraph equivalent:
            ComputationGraph g = net.toComputationGraph();

            boolean gradOKCG = GradientCheckUtil.checkGradients(g, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[]{in}, new INDArray[]{labels},
                    null, null, excludeParams);
            assertTrue(gradOKCG);

            TestUtils.testModelSerialization(g);
        }

    }
}
