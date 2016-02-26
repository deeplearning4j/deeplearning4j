package org.nd4j.linalg.util;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 2/26/16.
 */
public class NDArrayUtil {

    public static INDArray toNDArray(int[][] nums) {
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            double[] doubles = ArrayUtil.toDoubles(nums);
            INDArray create = Nd4j.create(doubles, new int[]{1, nums.length});
            return create;
        } else {
            float[] doubles = ArrayUtil.toFloats(nums);
            INDArray create = Nd4j.create(doubles, new int[]{1, nums.length});
            return create;
        }

    }

    public static INDArray toNDArray(int[] nums) {
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            double[] doubles = ArrayUtil.toDoubles(nums);
            INDArray create = Nd4j.create(doubles, new int[]{1, nums.length});
            return create;
        } else {
            float[] doubles = ArrayUtil.toFloats(nums);
            INDArray create = Nd4j.create(doubles, new int[]{1, nums.length});
            return create;
        }
    }


    public static int[] toInts(INDArray n) {
        if (n instanceof IComplexNDArray)
            throw new IllegalArgumentException("Unable to convert complex array");
        n = n.linearView();
        int[] ret = new int[n.length()];
        for (int i = 0; i < n.length(); i++)
            ret[i] = (int) n.getFloat(i);
        return ret;
    }

}
