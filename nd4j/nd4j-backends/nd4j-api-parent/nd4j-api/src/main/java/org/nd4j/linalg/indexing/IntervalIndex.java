package org.nd4j.linalg.indexing;

import com.google.common.primitives.Longs;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * And indexing representing
 * an interval
 *
 * @author Adam Gibson
 */
public class IntervalIndex implements INDArrayIndex {

    protected long begin, end;
    protected boolean inclusive;
    protected long stride = 1;
    protected long index = 0;
    protected long length = 0;

    /**
     *
     * @param inclusive whether to include the last number
     * @param stride the stride for the interval
     */
    public IntervalIndex(boolean inclusive, long stride) {
        this.inclusive = inclusive;
        this.stride = stride;

        this.length = (int) Math.abs((end - begin)) / stride;
    }

    @Override
    public long end() {
        return end;
    }

    @Override
    public long offset() {
        return begin;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public long stride() {
        return stride;
    }

    @Override
    public long current() {
        return index;
    }

    @Override
    public boolean hasNext() {
        return index < end();
    }

    @Override
    public long next() {
        long ret = index;
        index += stride;
        return ret;
    }


    @Override
    public void reverse() {
        long oldEnd = end;
        long oldBegin = begin;
        this.end = oldBegin;
        this.begin = oldEnd;
    }

    @Override
    public boolean isInterval() {
        return true;
    }

    @Override
    public void setInterval(boolean isInterval) {
       //no-op
    }

    @Override
    public void init(INDArray arr, long begin, int dimension) {
        if(begin < 0) {
            begin +=  arr.size(dimension);
        }

        this.begin = begin;
        this.index = begin;
        this.end = inclusive ? arr.size(dimension) + 1 : arr.size(dimension);
        for (long i = begin; i < end; i += stride) {
            length++;
        }
    }

    @Override
    public void init(INDArray arr, int dimension) {
        init(arr, 0, dimension);
    }


    @Override
    public void init(long begin, long end, long max) {
        if(begin < 0) {
            begin +=  max;
        }

        if(end < 0) {
            end +=  max;
        }
        this.begin = begin;
        this.index = begin;
        this.end = inclusive ? end + 1 : end;
        for (long i = begin; i < this.end; i += stride) {
            length++;
        }

    }

    @Override
    public void init(long begin, long end) {
        if(begin < 0 || end < 0)
            throw new IllegalArgumentException("Please pass in an array for negative indices. Unable to determine size for dimension otherwise");
        this.begin = begin;
        this.index = begin;
        this.end = inclusive ? end + 1 : end;
        for (long i = begin; i < this.end; i += stride) {
            length++;
        }

    }

    @Override
    public void reset() {

    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof IntervalIndex))
            return false;

        IntervalIndex that = (IntervalIndex) o;

        if (begin != that.begin)
            return false;
        if (end != that.end)
            return false;
        if (inclusive != that.inclusive)
            return false;
        if (stride != that.stride)
            return false;
        return index == that.index;

    }

    @Override
    public int hashCode() {
        int result = Longs.hashCode(begin);
        result = 31 * result + Longs.hashCode(end);
        result = 31 * result + (inclusive ? 1 : 0);
        result = 31 * result + Longs.hashCode(stride);
        result = 31 * result + Longs.hashCode(index);
        return result;
    }
}
