package org.deeplearning4j.nn.layers.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.graph.ShiftVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.samediff.testlayers.SameDiffDense;
import org.deeplearning4j.nn.layers.samediff.testlayers.SameDiffSimpleLambdaLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

@Slf4j
public class TestSameDiffLambda {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    @Test
    public void testSameDiffLamdaBasic(){
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.01))
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(5).nOut(5).activation(Activation.TANH).build(), "1")
                .addLayer("1", new SameDiffSimpleLambdaLayer(), "0")
                .addLayer("2", new OutputLayer.Builder().nIn(5).nOut(5).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "1")
                .setOutputs("2")
                .build();

        //Equavalent, not using SameDiff Lambda:
        ComputationGraphConfiguration confStd = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.01))
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(5).nOut(5).activation(Activation.TANH).build(), "1")
                .addVertex("1", new ShiftVertex(1.0), "0")
                .addVertex("2", new ScaleVertex(0.5), "1")
                .addLayer("3", new OutputLayer.Builder().nIn(5).nOut(5).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "2")
                .setOutputs("2")
                .build();

        ComputationGraph lambda = new ComputationGraph(conf);
        lambda.init();

        ComputationGraph std = new ComputationGraph(conf);
        std.init();

        lambda.setParams(std.params());

        INDArray in = Nd4j.rand(3,5);
        INDArray labels = TestUtils.randomOneHot(3, 5);
        DataSet ds = new DataSet(in, labels);

        INDArray outLambda = lambda.outputSingle(in);
        INDArray outStd = std.outputSingle(in);

        assertEquals(outLambda, outStd);

        double scoreLambda = lambda.score(ds);
        double scoreStd = std.score(ds);

        assertEquals(scoreStd, scoreLambda, 1e-6);

        for( int i=0; i<3; i++ ){
            lambda.fit(ds);
            std.fit(ds);

            String s = String.valueOf(i);
            assertEquals(s, std.params(), lambda.params());
            assertEquals(s, std.getFlattenedGradients(), lambda.getFlattenedGradients());
        }

        ComputationGraph loaded = TestUtils.testModelSerialization(lambda);
        outLambda = loaded.outputSingle(in);
        outStd = std.outputSingle(in);

        assertEquals(outStd, outLambda);
    }
}
