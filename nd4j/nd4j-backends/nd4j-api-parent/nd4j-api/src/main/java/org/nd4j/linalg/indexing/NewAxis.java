package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * New axis index.
 * Specified for wanting a new dimension
 * in an ndarray
 *
 * @author Adam Gibson
 */
public class NewAxis implements INDArrayIndex {
    @Override
    public long end() {
        return 0;
    }

    @Override
    public long offset() {
        return 0;
    }

    @Override
    public long length() {
        return 0;
    }

    @Override
    public long stride() {
        return 1;
    }

    @Override
    public long current() {
        return 0;
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public long next() {
        return 0;
    }


    @Override
    public void reverse() {

    }

    @Override
    public boolean isInterval() {
        return false;
    }

    @Override
    public void setInterval(boolean isInterval) {

    }

    @Override
    public void init(INDArray arr, long begin, int dimension) {

    }

    @Override
    public void init(INDArray arr, int dimension) {

    }

    @Override
    public void init(long begin, long end, long max) {

    }

    @Override
    public void init(long begin, long end) {

    }

    @Override
    public void reset() {

    }
}
