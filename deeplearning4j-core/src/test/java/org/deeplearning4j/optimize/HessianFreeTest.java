package org.deeplearning4j.optimize;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.sda.StackedDenoisingAutoEncoder;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 6/29/14.
 */
public class HessianFreeTest {

    private static Logger log = LoggerFactory.getLogger(HessianFreeTest.class);

    private DeepAutoEncoder a;
    private ClassPathResource r;
    private OptimizableByGradientValueMatrix o;

    @Before
    public void init() throws Exception {
        if(a == null) {
            ClassPathResource r = new ClassPathResource("gauss-vector.txt");
            DoubleMatrix d = DoubleMatrix.loadCSVFile(r.getFile().getAbsolutePath());
            assertEquals(2837314,d.length);

            int codeLayer = 3;

        /*
          Reduction of dimensionality with neural nets Hinton 2006
         */
            Map<Integer,Double> layerLearningRates = new HashMap<>();
            layerLearningRates.put(codeLayer,1e-1);
            RandomGenerator rng = new MersenneTwister(123);


            StackedDenoisingAutoEncoder dbn = new StackedDenoisingAutoEncoder.Builder()
                    .learningRateForLayer(layerLearningRates).constrainGradientToUnitNorm(false)
                    .hiddenLayerSizes(new int[]{1000, 500, 250, 30}).withRng(rng)
                    .activateForLayer(Collections.singletonMap(3, Activations.sigmoid())).useGaussNewtonVectorProductBackProp(true)
                    .numberOfInputs(784).sampleFromHiddenActivations(false).withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.HESSIAN_FREE)
                    .lineSearchBackProp(false).useRegularization(true).forceEpochs().weightInit(WeightInit.SI).outputLayerWeightInit(WeightInit.SI)
                    .withL2(0).lineSearchBackProp(true)
                    .withOutputActivationFunction(Activations.sigmoid())
                    .numberOfOutPuts(784).withOutputLossFunction(OutputLayer.LossFunction.RMSE_XENT)
                    .build();


            a = new DeepAutoEncoder.Builder().withEncoder(dbn).build();
            a.setParameters(d);
            assertEquals(a.pack(),d);


            r = new ClassPathResource("inputs.txt");
            DoubleMatrix miniBatch = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
            r = new ClassPathResource("labels.txt");
            DoubleMatrix labels = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());




            assertEquals(10, miniBatch.rows);
            assertEquals(10,labels.rows);
            assertEquals(784,miniBatch.columns);
            assertEquals(784,miniBatch.columns);
            assertEquals(miniBatch,labels);

            a.setInput(miniBatch);
            a.setLabels(labels);
            o = new BackPropROptimizer(a,1e-1,1000);

        }
    }


    @Test
    public void testHessianFree() throws Exception {
        StochasticHessianFree s = new StochasticHessianFree(o,1e-1,a);
        r = new ClassPathResource("grad.txt");
        DoubleMatrix grad = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        r = new ClassPathResource("x0.txt");
        DoubleMatrix x0 = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        r = new ClassPathResource("precon.txt");
        DoubleMatrix preCon = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        Pair<DoubleMatrix,DoubleMatrix> gradPrecon = a.getBackPropGradient2();
        DoubleMatrix gradTest = gradPrecon.getFirst();
        DoubleMatrix preContTest = gradPrecon.getSecond();
        DoubleMatrix chTest = DoubleMatrix.zeros(1,a.pack().length);
        log.info("Gradient sum " + grad.sum() + " with gradient test sum " + gradTest.sum());
        log.info("X0 sum " + x0.sum() + " with test " + chTest.sum());
        log.info("Precon sum " + preCon.sum() + " with test " + preContTest.sum());


        s.conjGradient(gradTest.neg(),chTest,preContTest,24);

    }


}
