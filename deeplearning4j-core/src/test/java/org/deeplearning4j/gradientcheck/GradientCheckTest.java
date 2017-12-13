package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.layers.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.deeplearning4j.gradientcheck.GradientCheckUtil.checkGradients;
import static org.junit.Assert.assertTrue;

/**
 * created by jingshu
 */

public class GradientCheckTest {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    private static Logger log = (Logger) LoggerFactory.getLogger(GradientCheckTest.class);

    @Test
    public void elementWiseMultiplicationLayerTest(){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).updater(new NoOp())
                .seed(12345L)
                .graphBuilder()
                .addInputs("features")
                .addLayer("dense",new DenseLayer.Builder().nIn(4).nOut(4)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(0, 1))
                        .activation(Activation.IDENTITY)
                        .build(),"features")
                .addLayer("elementWiseMul",new ElementWiseMultiplicationLayer.Builder().nIn(4).nOut(4)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(0, 1))
                        .activation(Activation.IDENTITY)
                        .build(),"dense")
                .addLayer("loss",new LossLayer.Builder(LossFunctions.LossFunction.COSINE_PROXIMITY).activation(Activation.IDENTITY). build(),"elementWiseMul")
                .setOutputs("loss")
                .pretrain(false).backprop(true).build();

        ComputationGraph netGraph = new ComputationGraph(conf);
        netGraph.init();

        if (PRINT_RESULTS) {
            System.out.println( netGraph.summary());
        }
        log.info("params before learning: "+netGraph.getLayer(1).paramTable());

        //Run a number of iterations of learning
//        manually make some pseudo data
//        the ides is simple: since we do a element wise multiplication layer (just a scaling), we want the cos sim is mainly decided by the fourth value, if everything runs well, we will get a large weight for the fourth value

        INDArray features = Nd4j.create(new double[][]{{1,2,3,4},{1,2,3,1},{1,2,3,0}});
        INDArray labels = Nd4j.create(new double[][]{{1,1,1,8},{1,1,1,2},{1,1,1,1}});

        netGraph.setInputs(features);
        netGraph.setLabels(labels);
        netGraph.computeGradientAndScore();
        double scoreBefore = netGraph.score();

        String msg;
        for (int epoch = 0; epoch < 5; epoch++)
            netGraph.fit(new INDArray[]{features},new INDArray[]{labels});
        netGraph.computeGradientAndScore();
        double scoreAfter = netGraph.score();
        //Can't test in 'characteristic mode of operation' if not learning
        msg = "elementWiseMultiplicationLayerTest() - score did not (sufficiently) decrease during learning - activationFn="
                + "Id" + ", lossFn=" + "Cos-sim" + ", outputActivation=" + "Id"
                + ", doLearningFirst=" + "true" + " (before=" + scoreBefore
                + ", scoreAfter=" + scoreAfter + ")";
        assertTrue(msg, scoreAfter < 0.8 * scoreBefore);

//        expectation in case linear regression(with only element wise multiplication layer): large weight for the fourth weight
        log.info("params after learning: "+netGraph.getLayer(1).paramTable());

        boolean gradOK = checkGradients(netGraph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[]{features}, new INDArray[]{labels});

        msg = "elementWiseMultiplicationLayerTest() - activationFn=" + "ID" + ", lossFn=" + "Cos-sim"
                + ", outputActivation=" + "Id" + ", doLearningFirst=" + "true";
        assertTrue(msg, gradOK);

        TestUtils.testModelSerialization(netGraph);

    }

}
