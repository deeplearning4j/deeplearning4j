package org.nd4j.linalg.indexing;

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
    public int[] indices() {
        return new int[0];
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
}
