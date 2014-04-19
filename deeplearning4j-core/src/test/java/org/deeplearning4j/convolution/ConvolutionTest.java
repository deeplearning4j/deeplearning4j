package org.deeplearning4j.convolution;

import static org.junit.Assert.*;
import static org.deeplearning4j.util.Convolution.*;

import org.deeplearning4j.util.Convolution;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 4/19/14.
 */
public class ConvolutionTest {

    private static Logger log = LoggerFactory.getLogger(ConvolutionTest.class);

    @Test
    public void testDiscreteFourierTransform() {

        DoubleMatrix rand = DoubleMatrix.rand(3);
        DoubleMatrix fourier = disceteFourierTransform(rand);
        DoubleMatrix inverse = inverseDisceteFourierTransform(rand);
        DoubleMatrix toAndFro = inverseDisceteFourierTransform(disceteFourierTransform(rand));
        assertEquals(true,toAndFro.distance1(rand) < 1e-1);
        log.info("Back and forth " + toAndFro);

    }

}
