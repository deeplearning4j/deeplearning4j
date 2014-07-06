package org.deeplearning4j.nn;

import org.jblas.DoubleMatrix;

/**
 * Interface for outputting  a value
 * relative to an output
 */
public interface Output {

    DoubleMatrix output(DoubleMatrix input);


}
