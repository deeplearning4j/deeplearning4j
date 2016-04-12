package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */ //static type checking used for checking if an index should be represented as all
public class NDArrayIndexEmpty  implements INDArrayIndex {
    @Override
    public int end() {
        return 0;
    }

    @Override
    public int offset() {
        return 0;
    }

    @Override
    public int length() {
        return 0;
    }

    @Override
    public int stride() {
        return 1;
    }

    @Override
    public int current() {
        return 0;
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public int next() {
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

    }
}
