package org.deeplearning4j.nn;

import org.deeplearning4j.nn.activation.Activations;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Test for weight init utils
 */
public class WeightInitUtilTest {

    private static Logger log = LoggerFactory.getLogger(WeightInitUtilTest.class);

    @Test
    public void testWeightIntSi() {
        DoubleMatrix si = WeightInitUtil.initWeights(784,1000,WeightInit.SI, Activations.sigmoid());
        DoubleMatrix si2 = WeightInitUtil.initWeights(784,1000,WeightInit.SI, Activations.rectifiedLinear());

    }

    @Test
    public void testWeightIntVi() {
        DoubleMatrix si = WeightInitUtil.initWeights(784,1000,WeightInit.VI, Activations.sigmoid());
        DoubleMatrix si2 = WeightInitUtil.initWeights(784,1000,WeightInit.VI, Activations.rectifiedLinear());

    }

}
