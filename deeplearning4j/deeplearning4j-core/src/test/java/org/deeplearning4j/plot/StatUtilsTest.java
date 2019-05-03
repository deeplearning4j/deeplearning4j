package org.deeplearning4j.plot;

import org.apache.commons.math3.util.FastMath;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class StatUtilsTest {

    @Test
    public void testBinarySearch() {
        INDArray distances = Nd4j.rand(50, 2);
        double desired_perplexity = 25.0;
        //distances = np.abs(distances.dot(distances.T))
        //            np.fill_diagonal(distances, 0.0)
        INDArray P = StatUtils.binary_search_perplexity(distances, desired_perplexity);

        //mean_perplexity = np.mean([np.exp(-np.sum(P[i] * np.log(P[i])))
        double sum = 0.0;
        INDArray logP = Nd4j.create(P.shape());
        for (int i = 0; i < P.rows(); ++i) {
            for (int j = 0; j < P.columns(); ++j) {
                double value = P.getRow(i).getDouble(j) * FastMath.log(P.getRow(i).getDouble(j));
                sum += value;
                logP.putScalar(i, j, FastMath.exp(FastMath.log(value)));
            }
        }

        INDArray expected_perplexity = Nd4j.mean(logP);
        assertEquals(expected_perplexity.getDouble(0), desired_perplexity, 1e-5);
    }
}
