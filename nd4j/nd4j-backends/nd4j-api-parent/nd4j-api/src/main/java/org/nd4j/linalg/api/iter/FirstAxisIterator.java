package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Iterator;

/**
 * @author Christian Weilbach
 */
public class FirstAxisIterator implements Iterator<Object> {
    private INDArray iterateOver;
    private int i = 0;


    /**
     *
     * @param iterateOver
     */
    public FirstAxisIterator(INDArray iterateOver) {
        this.iterateOver = iterateOver;
    }

    @Override
    public boolean hasNext() {
        return i < iterateOver.slices();
    }

    @Override
    public void remove() {

    }

    @Override
    public Object next() {
        INDArray s = iterateOver.slice(i++);
        if (s.isScalar()) {
            return s.getDouble(0);
        } else {
            return s;
        }
    }

}
