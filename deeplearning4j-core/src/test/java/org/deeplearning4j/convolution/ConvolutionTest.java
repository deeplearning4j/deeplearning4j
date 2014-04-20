package org.deeplearning4j.convolution;

import static org.junit.Assert.*;
import static org.deeplearning4j.util.Convolution.*;

import org.deeplearning4j.util.Convolution;
import org.jblas.ComplexDoubleMatrix;
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

        DoubleMatrix rand = new DoubleMatrix(new double[]{0.231312,0.17572,0.571717});
        ComplexDoubleMatrix fourier = complexDisceteFourierTransform(rand);
        ComplexDoubleMatrix inverse = complexInverseDisceteFourierTransform(rand);
        ComplexDoubleMatrix toAndFro = complexInverseDisceteFourierTransform(complexDisceteFourierTransform(rand));

        log.info("Back and forth " + toAndFro);

    }

}
