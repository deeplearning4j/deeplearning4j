package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class SpecifiedIndex implements INDArrayIndex {
    private int[] indexes;
    private int counter = 0;

    public SpecifiedIndex(int...indexes) {
        this.indexes = indexes;
    }

    @Override
    public int end() {
        return indexes[indexes.length - 1];
    }

    @Override
    public int offset() {
        return indexes[0];
    }

    @Override
    public int length() {
        return indexes.length;
    }

    @Override
    public int stride() {
        return 1;
    }

    @Override
    public int current() {
        return indexes[counter - 1];
    }

    @Override
    public boolean hasNext() {
        return counter < indexes.length;
    }

    @Override
    public int next() {
        return indexes[counter++];
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
    public void init(INDArray arr, int begin, int dimension) {

    }

    @Override
    public void init(INDArray arr, int dimension) {

    }

    @Override
    public void init(int begin, int end) {

    }

    @Override
    public void reset() {
        counter = 0;
    }
}
