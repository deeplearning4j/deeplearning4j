package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.Shape;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class INDArrayIterator implements Iterator<Double> {
    private INDArray iterateOver;
    private int i = 0;


    /**
     *
     * @param iterateOver
     */
    public INDArrayIterator(INDArray iterateOver) {
        this.iterateOver = iterateOver;
    }

    @Override
    public boolean hasNext() {
        return i < iterateOver.length();
    }

    @Override
    public Double next() {
        return iterateOver.getDouble(iterateOver.ordering() == 'c' ? Shape.ind2subC(iterateOver,i++) : Shape.ind2sub(iterateOver,i++));
    }
}
