package org.nd4j.linalg.api.ops;

/**
 * A broad cast op is one where a scalar
 * or less rank array
 * is broadcast to fill
 * a bigg er array.
 *
 * A typical broad cast operation would be adding a row to
 * each row in a matrix.
 *
 * @author Adam Gibson
 */
public interface BroadcastOp extends Op {

    /** Dimension to do the vector op along. Along dimension 1 for row vector ops,  along 0 for column vector ops */
    int[] getDimension();

    /** Set the dimension for the vector op. */
    void setDimension(int...dimension);


    /**
     * The length of the number of elements
     * in the broadcast
     * @return
     */
    int broadcastLength();

    /**
     * The shape of the
     * element to be broadcast
     * @return
     */
    int[] broadcastShape();

}
