package org.nd4j.linalg.api.ops;

public interface VectorOp extends Op {

    /** Dimension to do the vector op along. 0 for row vector ops, 1 for column vector ops */
    int getDimension();

    /** Set the dimension for the vector op. */
    void setDimension(int dimension);

}
