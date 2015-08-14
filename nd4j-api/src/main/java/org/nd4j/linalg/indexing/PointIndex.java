package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class PointIndex implements INDArrayIndex {
    private int point;
    private boolean notUsed = true;
    /**
     *
     * @param point
     */
    public PointIndex(int point) {
        this.point = point;
    }

    @Override
    public int end() {
        return point;
    }

    @Override
    public int offset() {
        return point;
    }

    @Override
    public int length() {
        return 1;
    }

    @Override
    public int stride() {
        return 1;
    }

    @Override
    public int current() {
        return point;
    }

    @Override
    public boolean hasNext() {
        return notUsed;
    }

    @Override
    public int next() {
        int ret =  point;
        notUsed = false;
        return ret;
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
        notUsed = false;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof PointIndex)) return false;

        PointIndex that = (PointIndex) o;

        if (point != that.point) return false;
        return notUsed == that.notUsed;

    }

    @Override
    public int hashCode() {
        int result = point;
        result = 31 * result + (notUsed ? 1 : 0);
        return result;
    }
}
