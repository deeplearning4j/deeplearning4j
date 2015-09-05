package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.api.shape.Shape;

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
    private char order = 'c';

    public NdIndexIterator(char order) {
        this.order = order;
    }


    /**
     *  Pass in the shape to iterate over.
     *  Defaults to c ordering
     * @param shape the shape to iterate over
     */
    public NdIndexIterator(int...shape) {
        this('c',shape);
    }


    /**
     *  Pass in the shape to iterate over
     * @param shape the shape to iterate over
     */
    public NdIndexIterator(char order,int...shape) {
        this.shape = ArrayUtil.copy(shape);
        this.length = ArrayUtil.prod(shape);
        this.order = order;
    }

    @Override
    public boolean hasNext() {
        return i < length;
    }



    @Override
    public int[] next() {
        switch(order) {
            case 'c':  return Shape.ind2subC(shape,i++,length);
            case 'f':  return Shape.ind2sub(shape, i++,length);
            default: throw new IllegalArgumentException("Illegal ordering " + order);
        }

    }

    @Override
    public void remove() {

    }
}
