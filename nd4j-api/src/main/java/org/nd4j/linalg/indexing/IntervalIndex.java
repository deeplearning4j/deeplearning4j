package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * And indexing representing
 * an interval
 *
 * @author Adam Gibson
 */
public class IntervalIndex implements INDArrayIndex {

    protected int begin,end;
    protected boolean inclusive;
    protected int stride = 1;
    protected int index = 0;
    /**
     *
     * @param inclusive whether to include the last number
     * @param stride the stride for the interval
     */
    public IntervalIndex(boolean inclusive, int stride) {
        this.inclusive = inclusive;
        this.stride = stride;
    }

    @Override
    public int end() {
        return end;
    }

    @Override
    public int offset() {
        return begin;
    }

    @Override
    public int length() {
        return (end - begin) / stride;
    }

    @Override
    public int stride() {
        return stride;
    }

    @Override
    public int current() {
        return index;
    }

    @Override
    public boolean hasNext() {
        return index < length();
    }

    @Override
    public int next() {
        int ret = index;
        index += stride;
        return  ret;
    }


    @Override
    public void reverse() {
        int oldEnd = end;
        int oldBegin = begin;
        this.end = oldBegin;
        this.begin = oldEnd;
    }

    @Override
    public boolean isInterval() {
        return true;
    }

    @Override
    public void setInterval(boolean isInterval) {

    }

    @Override
    public void init(INDArray arr, int begin, int dimension) {
        this.begin = begin;
        this.end = inclusive ? arr.size(dimension) + 1 : arr.size(dimension);
    }

    @Override
    public void init(INDArray arr, int dimension) {
        init(arr,0,dimension);
    }

    @Override
    public void init(int begin, int end) {
        this.begin = begin;
        this.end = inclusive ? end + 1 : end;

    }
}
