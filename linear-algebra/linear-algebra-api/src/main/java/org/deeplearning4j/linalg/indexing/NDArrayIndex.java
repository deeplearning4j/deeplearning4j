package org.deeplearning4j.linalg.indexing;

import org.deeplearning4j.linalg.util.ArrayUtil;

/**
 * NDArray indexing
 *
 * @author Adam Gibson
 */
public class NDArrayIndex {

    private int[] indices;


    public NDArrayIndex(int[] indices) {

        this.indices = indices;
    }



    public int[] indices() {
        return indices;
    }

    public void reverse() {
        ArrayUtil.reverse(indices);
    }


    public static NDArrayIndex interval(int begin,int end) {
        return new NDArrayIndex(ArrayUtil.range(begin,end));
    }

}
