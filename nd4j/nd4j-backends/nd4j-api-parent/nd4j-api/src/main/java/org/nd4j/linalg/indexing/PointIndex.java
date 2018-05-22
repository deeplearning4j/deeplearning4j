package org.nd4j.linalg.indexing;

import com.google.common.primitives.Longs;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class PointIndex implements INDArrayIndex {
    private long point;
    private boolean notUsed = true;

    /**
     *
     * @param point
     */
    public PointIndex(long point) {
        this.point = point;
    }

    @Override
    public long end() {
        return point;
    }

    @Override
    public long offset() {
        return point;
    }

    @Override
    public long length() {
        return 1;
    }

    @Override
    public long stride() {
        return 1;
    }

    @Override
    public long current() {
        return point;
    }

    @Override
    public boolean hasNext() {
        return notUsed;
    }

    @Override
    public long next() {
        long ret = point;
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
        notUsed = false;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof PointIndex))
            return false;

        PointIndex that = (PointIndex) o;

        if (point != that.point)
            return false;
        return notUsed == that.notUsed;

    }

    @Override
    public int hashCode() {
        int result = Longs.hashCode(point);
        result = 31 * result + (notUsed ? 1 : 0);
        return result;
    }
}
