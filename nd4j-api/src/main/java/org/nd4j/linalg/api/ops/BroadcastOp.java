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
