package org.deeplearning4j.arbiter.optimize.distribution;

import org.apache.commons.math3.distribution.RealDistribution;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestLogUniform {

    @Test
    public void testSimple(){

        double min = 0.5;
        double max = 3;

        double logMin = Math.log(min);
        double logMax = Math.log(max);

        RealDistribution rd = new LogUniformDistribution(min, max);

        for(double d = 0.1; d<= 3.5; d+= 0.1){
            double density = rd.density(d);
            double cumulative = rd.cumulativeProbability(d);
            double dExp;
            double cumExp;
            if(d < min){
                dExp = 0;
                cumExp = 0;
            } else if( d > max){
                dExp = 0;
                cumExp = 1;
            } else {
                dExp = 1.0 / (d * (logMax-logMin));
                cumExp = (Math.log(d) - logMin) / (logMax - logMin);
            }

            assertTrue(dExp >= 0);
            assertTrue(cumExp >= 0);
            assertTrue(cumExp <= 1.0);
            assertEquals(dExp, density, 1e-5);
            assertEquals(cumExp, cumulative, 1e-5);
        }

        rd.reseedRandomGenerator(12345);
        for( int i=0; i<100; i++ ){
            double d = rd.sample();
            assertTrue(d >= min);
            assertTrue(d <= max);
        }
    }

}
