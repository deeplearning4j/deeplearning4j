package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.util.ArrayUtil;

import java.util.Iterator;

/**
 * Created by agibsonccc on 9/15/15.
 */
public class FlatIterator implements Iterator<int[]> {

    private int[] shape;
    private int runningDimension;
    private int[] currentCoord;
    private int length;
    private int current = 0;

    public FlatIterator(int[] shape) {
        this.shape = shape;
        this.currentCoord = new int[shape.length];
        length = ArrayUtil.prod(shape);
    }

    @Override
    public void remove() {

    }

    @Override
    public boolean hasNext() {
        return current < length;
    }

    @Override
    public int[] next() {
        if (currentCoord[runningDimension] == shape[runningDimension]) {
            runningDimension--;
            currentCoord[runningDimension] = 0;
            if (runningDimension < shape.length) {

            }
        } else {
            //bump to the next coordinate
            currentCoord[runningDimension]++;
        }
        current++;
        return currentCoord;
    }
}
