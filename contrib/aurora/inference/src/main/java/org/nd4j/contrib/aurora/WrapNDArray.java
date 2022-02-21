package org.nd4j.contrib.aurora;

import org.nd4j.linalg.api.ndarray.INDArray;

public class WrapNDArray {

    public INDArray arr;

    public WrapNDArray(INDArray arr) {
        this.arr = arr;
    }

    @Override
    public boolean equals(Object o) {
        WrapNDArray w = (WrapNDArray) o;
        return w.arr != null && (w.arr.getId() == this.arr.getId());
    }

    @Override
    public int hashCode() {
        return (int) (this.arr.getId() % Integer.MAX_VALUE);
    }

}
