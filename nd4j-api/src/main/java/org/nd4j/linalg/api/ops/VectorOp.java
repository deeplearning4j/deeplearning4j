package org.nd4j.linalg.api.ops;

/** A VectorOp is an op where op that is applied to each 1d vector
 * Examples are add/sub/mul/div/copy for each row or for each column
 */
public interface VectorOp extends BroadcastOp {

    /** Dimension to do the vector op along. Along dimension 1 for row vector ops,  along 0 for column vector ops */
    int getDimension();

    /** Set the dimension for the vector op. */
    void setDimension(int dimension);

}
