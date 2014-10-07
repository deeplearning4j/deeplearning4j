package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;
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
     * Create from a matrix. The rows are the indices
     * The columns are the individual element in each ndarrayindex
     * @param index the matrix to get indices from
     * @return the indices to get
     */
    public static NDArrayIndex[] create(INDArray index) {
        assert index.isMatrix();
        NDArrayIndex[] ret = new NDArrayIndex[index.rows()];
        for(int i = 0; i < index.rows(); i++) {
            INDArray row = index.getRow(i);
            int[] nums = new int[index.getRow(i).columns()];
            for(int j = 0; j < row.columns(); j++) {
                nums[j] = (int) row.get(j);
            }

            NDArrayIndex idx = new NDArrayIndex(nums);
            ret[i]  = idx;

        }


        return ret;
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
        assert begin <= end : "Beginning index in range must be less than end";
        return new NDArrayIndex(ArrayUtil.range(begin,inclusive ?  end + 1 : end));
    }


}
