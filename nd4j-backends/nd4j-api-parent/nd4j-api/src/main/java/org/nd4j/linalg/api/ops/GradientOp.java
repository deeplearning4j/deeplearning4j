package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A gradient op represents
 * a jacobian operation
 *
 * @author Adam Gibson
 */
public  interface GradientOp extends  Op {

    /**
     * The array
     * to the gradient with respect to
     * @return
     */
    INDArray wrt();

}
