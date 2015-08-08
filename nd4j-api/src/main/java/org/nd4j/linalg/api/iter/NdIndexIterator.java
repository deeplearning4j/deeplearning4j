package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.Shape;

import java.util.Iterator;

/**
 * Iterates and returns int arrays
 * over a particular shape.
 *
 * This iterator starts at zero and increments
 * the shape until each item in the "position"
 * hits the current shape
 *
 * @author Adam Gibson
 */
public class NdIndexIterator implements Iterator<int[]> {
    private int length = -1;
    private int i = 0;
    private int[] shape;


    /**
     *  Pass in the shape to iterate over
     * @param shape the shape to iterate over
     */
    public NdIndexIterator(int...shape) {
        this.shape = ArrayUtil.copy(shape);
        this.length = ArrayUtil.prod(shape);
    }

    @Override
    public boolean hasNext() {
        return i < length;
    }



    @Override
    public int[] next() {
        return Shape.ind2subC(shape,i++);
    }

    @Override
    public void remove() {

    }
}
