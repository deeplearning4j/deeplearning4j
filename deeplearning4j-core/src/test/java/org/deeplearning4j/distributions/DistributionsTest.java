package org.deeplearning4j.distributions;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.junit.Test;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

/**
 * @author Adam Gibson
 */
public class DistributionsTest {

    private RandomGenerator rng = new MersenneTwister(123);
    private static Logger log = LoggerFactory.getLogger(DistributionsTest.class);
    @Test
    public void uniformTest() {

        RealDistribution dist = Distributions.uniform(rng,4);
        int numSamples = 10;
        double avg = 0.0;
        for(int i = 0;i < numSamples; i++)
            avg += dist.sample();

       avg /= numSamples;

        log.info("Avg " + avg);


    }

}
