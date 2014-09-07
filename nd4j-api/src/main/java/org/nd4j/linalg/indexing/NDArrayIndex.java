package org.nd4j.linalg.indexing;

import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 * NDArray indexing
 *
 * @author Adam Gibson
 */
public class NDArrayIndex {

    private int[] indices = new int[1];


    public NDArrayIndex(int[] indices) {
        if(indices.length > 0)
            this.indices = indices;
        else
            this.indices = new int[1];

    }



    public int end() {
        return indices[indices.length - 1];
    }

    public int offset() {
        if(indices.length < 1)
            return 0;
        return indices[0];
    }


    /**
     * Returns the length of the indices
     * @return the length of the range
     */
    public int length() {
        if(indices.length < 1)
            return 0;
        return indices[indices.length - 1] - indices[0];
    }

    public int[] indices() {
        return indices;
    }

    public void reverse() {
        ArrayUtil.reverse(indices);
    }


    @Override
    public String toString() {
        return "NDArrayIndex{" +
                "indices=" + Arrays.toString(indices) +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NDArrayIndex)) return false;

        NDArrayIndex that = (NDArrayIndex) o;

        if (!Arrays.equals(indices, that.indices)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(indices);
    }

    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     * @param begin the begin
     * @param end the end index
     * @return the interval
     */
    public static NDArrayIndex interval(int begin,int end) {
        return interval(begin,end,false);
    }


    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     * @param begin the begin
     * @param end the end index
     * @param inclusive whether the end should be inclusive or not
     * @return the interval
     */
    public static NDArrayIndex interval(int begin,int end,boolean inclusive) {
        return new NDArrayIndex(ArrayUtil.range(begin,inclusive ?  end + 1 : end));
    }


}
