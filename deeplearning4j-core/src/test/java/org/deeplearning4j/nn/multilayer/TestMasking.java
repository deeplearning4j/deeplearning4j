package org.deeplearning4j.nn.multilayer;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.gradientcheck.LossFunctionGradientCheck;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Not;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.*;

import java.util.Collections;

import static org.junit.Assert.*;

/**
 * Created by Alex on 20/01/2017.
 */
public class TestMasking extends BaseDL4JTest {

    static {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void checkMaskArrayClearance() {
        for (boolean tbptt : new boolean[] {true, false}) {
            //Simple "does it throw an exception" type test...
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list()
                            .layer(0, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                            .activation(Activation.IDENTITY).nIn(1).nOut(1).build())
                            .backpropType(tbptt ? BackpropType.TruncatedBPTT : BackpropType.Standard)
                            .tBPTTForwardLength(8).tBPTTBackwardLength(8).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            DataSet data = new DataSet(Nd4j.linspace(1, 10, 10).reshape(1, 1, 10),
                            Nd4j.linspace(2, 20, 10).reshape(1, 1, 10), Nd4j.ones(10), Nd4j.ones(10));

            net.fit(data);
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }


            net.fit(data.getFeatures(), data.getLabels(), data.getFeaturesMaskArray(), data.getLabelsMaskArray());
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }

            DataSetIterator iter = new ExistingDataSetIterator(Collections.singletonList(data).iterator());
            net.fit(iter);
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }
        }
    }

    @Test
    public void testPerOutputMaskingMLN() {
        //Idea: for per-output masking, the contents of the masked label entries should make zero difference to either
        // the score or the gradients

        int nIn = 6;
        int layerSize = 4;

        INDArray mask1 = Nd4j.create(new double[] {1, 0, 0, 1, 0});
        INDArray mask3 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {0, 1, 0, 1, 0}, {1, 0, 0, 1, 1}});
        INDArray[] labelMasks = new INDArray[] {mask1, mask3};

        ILossFunction[] lossFunctions = new ILossFunction[] {new LossBinaryXENT(),
                        //                new LossCosineProximity(),    //Doesn't support per-output masking, as it doesn't make sense for cosine proximity
                        new LossHinge(), new LossKLD(), new LossKLD(), new LossL1(), new LossL2(), new LossMAE(),
                        new LossMAE(), new LossMAPE(), new LossMAPE(),
                        //                new LossMCXENT(),             //Per output masking on MCXENT+Softmax: not yet supported
                        new LossMCXENT(), new LossMSE(), new LossMSE(), new LossMSLE(), new LossMSLE(),
                        new LossNegativeLogLikelihood(), new LossPoisson(), new LossSquaredHinge()};

        Activation[] act = new Activation[] {Activation.SIGMOID, //XENT
                        //                Activation.TANH,
                        Activation.TANH, //Hinge
                        Activation.SIGMOID, //KLD
                        Activation.SOFTMAX, //KLD + softmax
                        Activation.TANH, //L1
                        Activation.TANH, //L2
                        Activation.TANH, //MAE
                        Activation.SOFTMAX, //MAE + softmax
                        Activation.TANH, //MAPE
                        Activation.SOFTMAX, //MAPE + softmax
                        //                Activation.SOFTMAX, //MCXENT + softmax: see comment above
                        Activation.SIGMOID, //MCXENT + sigmoid
                        Activation.TANH, //MSE
                        Activation.SOFTMAX, //MSE + softmax
                        Activation.SIGMOID, //MSLE - needs positive labels/activations (due to log)
                        Activation.SOFTMAX, //MSLE + softmax
                        Activation.SIGMOID, //NLL
                        Activation.SIGMOID, //Poisson
                        Activation.TANH //Squared hinge
        };

        for (INDArray labelMask : labelMasks) {

            val minibatch = labelMask.size(0);
            val nOut = labelMask.size(1);

            for (int i = 0; i < lossFunctions.length; i++) {
                ILossFunction lf = lossFunctions[i];
                Activation a = act[i];


                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new NoOp())
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).seed(12345)
                                .list()
                                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                                .build())
                                .layer(1, new OutputLayer.Builder().nIn(layerSize).nOut(nOut).lossFunction(lf)
                                                .activation(a).build())
                                .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                net.setLayerMaskArrays(null, labelMask);
                INDArray[] fl = LossFunctionGradientCheck.getFeaturesAndLabels(lf, minibatch, nIn, nOut, 12345);
                INDArray features = fl[0];
                INDArray labels = fl[1];

                net.setInput(features);
                net.setLabels(labels);

                net.computeGradientAndScore();
                double score1 = net.score();
                INDArray grad1 = net.gradient().gradient();

                //Now: change the label values for the masked steps. The

                INDArray maskZeroLocations = Nd4j.getExecutioner().execAndReturn(new Not(labelMask.dup()));
                INDArray rand = Nd4j.rand(maskZeroLocations.shape()).muli(0.5);

                INDArray newLabels = labels.add(rand.muli(maskZeroLocations)); //Only the masked values are changed

                net.setLabels(newLabels);
                net.computeGradientAndScore();

                assertNotEquals(labels, newLabels);

                double score2 = net.score();
                INDArray grad2 = net.gradient().gradient();

                assertEquals(score1, score2, 1e-6);
                assertEquals(grad1, grad2);



                //Do the same for CompGraph
                ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder().updater(new NoOp())
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).seed(12345)
                                .graphBuilder().addInputs("in")
                                .addLayer("0", new DenseLayer.Builder().nIn(nIn).nOut(layerSize)
                                                .activation(Activation.TANH).build(), "in")
                                .addLayer("1", new OutputLayer.Builder().nIn(layerSize).nOut(nOut).lossFunction(lf)
                                                .activation(a).build(), "0")
                                .setOutputs("1").build();

                ComputationGraph graph = new ComputationGraph(conf2);
                graph.init();

                graph.setLayerMaskArrays(null, new INDArray[] {labelMask});

                graph.setInputs(features);
                graph.setLabels(labels);
                graph.computeGradientAndScore();

                double gScore1 = graph.score();
                INDArray gGrad1 = graph.gradient().gradient();

                graph.setLayerMaskArrays(null, new INDArray[] {labelMask});
                graph.setInputs(features);
                graph.setLabels(newLabels);
                graph.computeGradientAndScore();

                double gScore2 = graph.score();
                INDArray gGrad2 = graph.gradient().gradient();

                assertEquals(gScore1, gScore2, 1e-6);
                assertEquals(gGrad1, gGrad2);
            }
        }
    }

    @Test
    public void testCompGraphEvalWithMask() {
        int minibatch = 3;
        int layerSize = 6;
        int nIn = 5;
        int nOut = 4;

        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder().updater(new NoOp())
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).seed(12345)
                        .graphBuilder().addInputs("in")
                        .addLayer("0", new DenseLayer.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                        .build(), "in")
                        .addLayer("1", new OutputLayer.Builder().nIn(layerSize).nOut(nOut)
                                        .lossFunction(LossFunctions.LossFunction.XENT).activation(Activation.SIGMOID)
                                        .build(), "0")
                        .setOutputs("1").build();

        ComputationGraph graph = new ComputationGraph(conf2);
        graph.init();

        INDArray f = Nd4j.create(minibatch, nIn);
        INDArray l = Nd4j.create(minibatch, nOut);
        INDArray lMask = Nd4j.ones(minibatch, nOut);

        DataSet ds = new DataSet(f, l, null, lMask);
        DataSetIterator iter = new ExistingDataSetIterator(Collections.singletonList(ds).iterator());

        EvaluationBinary eb = new EvaluationBinary();
        graph.doEvaluation(iter, eb);
    }
}
