package org.deeplearning4j.nn;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.sda.StackedDenoisingAutoEncoder;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Gauss vector tests
 */
public class GaussVectorTests {

    private static Logger log = LoggerFactory.getLogger(GaussVectorTests.class);

    private DeepAutoEncoder a;
    private ClassPathResource r;

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


        }
    }

    @Test
    public void testGaussVector() throws Exception {




        List<DoubleMatrix> forward = a.feedForward();
        for(int i = 0; i < forward.size(); i++) {
            r = new ClassPathResource("acts-" + i + ".txt");
            DoubleMatrix realAct = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
            DoubleMatrix act1 = forward.get(i);
            log.info("Sums for " + i + " real is " + realAct.sum() + " with the test being " + act1.sum());


        }

        List<Pair<DoubleMatrix,DoubleMatrix>> deltas = a.computeDeltas2();
        for(int i = 0; i < deltas.size(); i++) {
            r = new ClassPathResource("delta-" + i + ".txt");
            DoubleMatrix delta = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
            DoubleMatrix deltaTest = deltas.get(i).getFirst();
            log.info("Delta " + i + " real " + delta.sum() + " test delta " + deltaTest.sum());

            r = new ClassPathResource("delta2-" + i + ".txt");
            DoubleMatrix delta2 = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
            DoubleMatrix delta2Test = deltas.get(i).getSecond();

            log.info("Delta 2 " + i + " real " + delta2.sum() + " test delta " + delta2Test.sum());

        }



        r = new ClassPathResource("grad.txt");
        DoubleMatrix grad = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        r = new ClassPathResource("precon.txt");
        DoubleMatrix preCon = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        Pair<DoubleMatrix,DoubleMatrix> backProp = a.getBackPropGradient2();
        log.info("Gradient sum loaded " + grad.sum() + " test sum was " + backProp.getFirst().sum());
        log.info("Precon sum loaded " + preCon.sum() + " test sum was " + backProp.getSecond().sum());


    }

    @Test
    public void testGv() throws Exception {
        DoubleMatrix pack = a.pack();
        DoubleMatrix ch = DoubleMatrix.zeros(1,pack.length);
        List<DoubleMatrix> rFeedForward = a.feedForwardR(ch);
        for(int i = 0; i < rFeedForward.size(); i++) {
            r = new ClassPathResource("ract-" + i + ".txt");
            DoubleMatrix real = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
            DoubleMatrix test = rFeedForward.get(i);
            log.info("Sum for r " + i + " is "  + real.sum() + " and " + test.sum());


        }

        ch = DoubleMatrix.zeros(1,pack.length);
        List<DoubleMatrix> backward = a.computeDeltasR(ch);

        for(int i = 0; i < backward.size(); i++) {
            r = new ClassPathResource("rdelta-" + i + ".txt");
            DoubleMatrix real = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
            DoubleMatrix test = backward.get(i);

            log.info("Sum for r backward prop " + i + " real " + real.sum() + " test " + test.sum());


        }

        r = new ClassPathResource("gv.txt");
        DoubleMatrix gv = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        DoubleMatrix gvTest = a.getBackPropRGradient(ch);
        log.info("Gv sum is " + gv.sum() + " with test gv sum " + gvTest.sum());





    }

    @Test
    public void testRy() throws Exception {
        r = new ClassPathResource("r.txt");
        DoubleMatrix rMatrix = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        r = new ClassPathResource("y.txt");
        DoubleMatrix yMatrix = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        log.info(" R and y multiplied sum " + rMatrix.mul(yMatrix).sum());
        r = new ClassPathResource("b.txt");
        DoubleMatrix b = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        r = new ClassPathResource("x0.txt");
        DoubleMatrix x0 = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        r = new ClassPathResource("gv.txt");
        DoubleMatrix gv = DoubleMatrix.loadAsciiFile(r.getFile().getAbsolutePath());
        assertEquals(x0.length,a.pack().length);
        assertEquals(b.length,a.pack().length);

        

        DoubleMatrix gvTest = a.getBackPropRGradient(x0);
        log.info("Gv sum is " + gv.sum() + " with test gv sum " + gvTest.sum());

        DoubleMatrix rTest = gvTest.sub(b);
        log.info("R real sum " + rMatrix.sum() + " with r test " + rTest.sum());




    }



}
