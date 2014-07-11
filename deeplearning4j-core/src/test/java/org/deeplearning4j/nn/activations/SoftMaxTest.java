package org.deeplearning4j.nn.activations;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.WeightInitUtil;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Test both kinds of activations on softmax
 * @author Adam Gibson
 */
public class SoftMaxTest {

    private static Logger log = LoggerFactory.getLogger(SoftMaxTest.class);

    @Test
    public void testSoftMax() {
        DoubleMatrix weights = WeightInitUtil.initWeights(10, 20, WeightInit.VI, Activations.softmax());
        DoubleMatrix rand = MatrixUtil.randDouble(20, 10, 0, 1, new MersenneTwister(123));
        DoubleMatrix randRows = MatrixUtil.randDouble(10,20, 0, 1, new MersenneTwister(123));
        //column wise features
        DoubleMatrix mult = rand.mmul(weights);
        //row wise features


        ActivationFunction softMaxColumns = Activations.softmax();

        DoubleMatrix activation = softMaxColumns.apply(mult);
        double columnSum = activation.sum() / activation.rows;
        double diffFrom1 = Math.abs(1 - columnSum);
        assertEquals(true,diffFrom1 <= 1e-1);


        DoubleMatrix rowWiseActivation = randRows.mmul(weights.transpose());
        ActivationFunction softMaxRows = Activations.softMaxRows();

        DoubleMatrix softMaxRowResult =   softMaxRows.apply(rowWiseActivation);
        double rowSum = softMaxRowResult.sum() / softMaxRowResult.columns;
        double diffFrom1Row = Math.abs(1 - rowSum);
        assertEquals(true,diffFrom1Row <= 1e-1);



    }

}
