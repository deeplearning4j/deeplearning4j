package org.deeplearning4j.linalg.util;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;

/**
 * Feature matrix related utils
 */
public class FeatureUtil {
    /**
     * Creates an out come vector from the specified inputs
     * @param index the index of the label
     * @param numOutcomes the number of possible outcomes
     * @return a binary label matrix used for supervised learning
     */
    public static INDArray toOutcomeVector(int index,int numOutcomes) {
        int[] nums = new int[numOutcomes];
        nums[index] = 1;
        return ArrayUtil.toNDArray(nums);
    }



    /**
     * Creates an out come vector from the specified inputs
     * @param index the index of the label
     * @param numOutcomes the number of possible outcomes
     * @return a binary label matrix used for supervised learning
     */
    public static INDArray toOutcomeMatrix(int[] index,int numOutcomes) {
        INDArray ret = NDArrays.create(index.length,numOutcomes);
        for(int i = 0; i < ret.rows(); i++) {
            int[] nums = new int[numOutcomes];
            nums[index[i]] = 1;
            ret.putRow(i, ArrayUtil.toNDArray(nums));
        }

        return ret;
    }

    public static  void normalizeMatrix(INDArray toNormalize) {
        INDArray columnMeans = toNormalize.mean(0);
        toNormalize.subiRowVector(columnMeans);
        INDArray std = toNormalize.std(0);
        std.addi(NDArrays.scalar(1e-6));
        toNormalize.diviRowVector(std);
    }

    /**
     * Divides each row by its max
     *
     * @param toScale the matrix to divide by its row maxes
     */
    public static void scaleByMax(INDArray toScale) {
        INDArray scale = toScale.max(1);
        for (int i = 0; i < toScale.rows(); i++) {
            float scaleBy = scale.get(i);
            toScale.putRow(i, toScale.getRow(i).divi(scaleBy));
        }
    }




}
