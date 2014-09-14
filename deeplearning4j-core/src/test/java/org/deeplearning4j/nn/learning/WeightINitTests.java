package org.deeplearning4j.nn.learning;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.WeightInitUtil;
import org.junit.Test;
import org.nd4j.linalg.api.activation.Activations;

/**
 * Created by agibsonccc on 9/13/14.
 */
public class WeightINitTests {

    @Test
    public void testSi() {
        WeightInitUtil.initWeights(1,2, WeightInit.VI, Activations.linear(), Distributions.normal(new MersenneTwister(123),1));
    }

}
