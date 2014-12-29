package org.deeplearning4j.distributions;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.junit.Test;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;
import static org.junit.Assert.assertEquals;

/**
 * @author Adam Gibson
 */
public class DistributionsTest {

    private RandomGenerator rng = new MersenneTwister(123);
    private static Logger LOG = LoggerFactory.getLogger(DistributionsTest.class);

    @Test
    public void uniformTest() {

        // generate random numbers in the range Unif[-4, 4]
        // Note: Expectation of this distribution is 0
        RealDistribution dist = Distributions.uniform(rng, 4);
        final double MEAN = 0.0;

        int numSamples = 10000000;
        double avg = 0.0;
        for(int i = 0; i < numSamples; i++)
            avg += dist.sample();
        avg /= numSamples;
        LOG.info("sample mean = " + avg);
        assertEquals("sample mean not converging to actual", MEAN, avg, 1E-3);
    }
}
