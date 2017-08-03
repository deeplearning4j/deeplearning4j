package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * Tests for memory-related stuff in gradients accumulator
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class EncodedGradientsAccumulatorTest {


    @Test
    public void testMemoryLimits1() throws Exception {
        int numParams = 100000;

        EncodedGradientsAccumulator accumulator = new EncodedGradientsAccumulator(2, new EncodingHandler(1e-3), 1000000, 2, null);



    }


    @Test
    public void testEncodingLimits1() throws Exception {
        int numParams = 100000;

        EncodingHandler handler = new EncodingHandler(1e-3);

        for (int e = 10; e < numParams / 5; e++) {
            INDArray encoded = handler.encodeUpdates(getGradients(numParams, e, 2e-3));

          //  log.info("enc len: {}", encoded.data().length());

            int encFormat = encoded.data().getInt(3);

            assertTrue("Failed for E = " + e + "; Format: " + encFormat + "; Length: " + encoded.data().length(), encoded.data().length() < numParams / 16 + 6);
        }
    }


  protected INDArray getGradients(int length, int numPositives, double value) {
        INDArray grad = Nd4j.create(length);

        for (int i = 0; i < numPositives; i++) {
            grad.putScalar(i, value);
        }

        return grad;
    }
}