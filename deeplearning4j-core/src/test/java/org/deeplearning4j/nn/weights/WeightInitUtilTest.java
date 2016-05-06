package org.deeplearning4j.nn.weights;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;



import static org.junit.Assert.*;

/**
 * Created by nyghtowl on 11/14/15.
 */
public class WeightInitUtilTest {
    protected int[] shape = new int[]{2, 2};
    protected Distribution dist = Distributions.createDistribution(new GaussianDistribution(0.0, 0.1));

    @Before
    public void doBefore(){
        Nd4j.getRandom().setSeed(123);
    }

    @Test
    public void testDistribution(){
        INDArray weightsActual = WeightInitUtil.initWeights(shape, WeightInit.DISTRIBUTION, dist);

        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected = dist.sample(shape);

        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    public void testNormalize(){
        INDArray weightsActual = WeightInitUtil.initWeights(shape, WeightInit.NORMALIZED, dist);

        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected =  Nd4j.rand('f',shape);
        weightsExpected.subi(0.5).divi(shape[0]);

        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    public void testRelu(){
        INDArray weightsActual = WeightInitUtil.initWeights(shape, WeightInit.RELU, dist);

        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected = Nd4j.randn('f',shape).muli(FastMath.sqrt(2.0 / shape[0]));

        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    public void testSize(){
        INDArray weightsActual = WeightInitUtil.initWeights(shape, WeightInit.SIZE, dist);

        // expected calculation
        Nd4j.getRandom().setSeed(123);
        double min = -4.0 * Math.sqrt(6.0 / (double) (shape[0] + shape[1]));
        double max = 4.0 * Math.sqrt(6.0 / (double) (shape[0] + shape[1]));
        INDArray weightsExpected = Nd4j.rand(shape, Nd4j.getDistributions().createUniform(min,max));

        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    public void testUniform(){
        INDArray weightsActual = WeightInitUtil.initWeights(shape, WeightInit.UNIFORM, dist);

        // expected calculation
        Nd4j.getRandom().setSeed(123);
        double a = 1/(double) shape[0];
        INDArray weightsExpected = Nd4j.rand('f',shape).muli(2*a).subi(a);

        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    public void testVI(){
        INDArray weightsActual = WeightInitUtil.initWeights(shape, WeightInit.VI, dist);

        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected = Nd4j.rand('f',shape);
        int numValues = shape[0] + shape[1];
        double r = Math.sqrt(6) / Math.sqrt(numValues + 1);
        weightsExpected.muli(2).muli(r).subi(r);

        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    public void testXavier(){
        INDArray weightsActual = WeightInitUtil.initWeights(shape, WeightInit.XAVIER, dist);

        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected = Nd4j.randn('f',shape);
        weightsExpected.divi(FastMath.sqrt(shape[0] + shape[1]));

        assertEquals(weightsExpected, weightsActual);
    }


    @Test
    public void testZero(){
        INDArray weightsActual = WeightInitUtil.initWeights(shape, WeightInit.ZERO, dist);

        // expected calculation
        INDArray weightsExpected = Nd4j.create(shape);

        assertEquals(weightsExpected, weightsActual);
    }


}
