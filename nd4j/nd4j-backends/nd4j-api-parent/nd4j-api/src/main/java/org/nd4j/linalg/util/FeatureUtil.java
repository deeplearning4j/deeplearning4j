/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.util;

import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Feature matrix related jcuda.utils
 */
public class FeatureUtil {
    /**
     * Creates an out come vector from the specified inputs
     *
     * @param index       the index of the label
     * @param numOutcomes the number of possible outcomes
     * @return a binary label matrix used for supervised learning
     */
    public static INDArray toOutcomeVector(long index, long numOutcomes) {
        if (index > Integer.MAX_VALUE || numOutcomes > Integer.MAX_VALUE)
            throw new UnsupportedOperationException();

        val nums = new int[(int) numOutcomes];
        nums[(int) index] = 1;
        return NDArrayUtil.toNDArray(nums);
    }


    /**
     * Creates an out come vector from the specified inputs
     *
     * @param index       the index of the label
     * @param numOutcomes the number of possible outcomes
     * @return a binary label matrix used for supervised learning
     */
    public static INDArray toOutcomeMatrix(int[] index, long numOutcomes) {
        INDArray ret = Nd4j.create(index.length, numOutcomes);
        for (int i = 0; i < ret.rows(); i++) {
            int[] nums = new int[(int) numOutcomes];
            nums[index[i]] = 1;
            ret.putRow(i, NDArrayUtil.toNDArray(nums));
        }

        return ret;
    }

    public static void normalizeMatrix(INDArray toNormalize) {
        INDArray columnMeans = toNormalize.mean(0);
        toNormalize.subiRowVector(columnMeans);
        INDArray std = toNormalize.std(0);
        std.addi(Nd4j.scalar(1e-12));
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
            double scaleBy = scale.getDouble(i);
            toScale.putRow(i, toScale.getRow(i).divi(scaleBy));
        }
    }


    /**
     * Scales the ndarray columns
     * to the given min/max values
     *
     * @param min the minimum number
     * @param max the max number
     */
    public static void scaleMinMax(double min, double max, INDArray toScale) {
        //X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) X_scaled = X_std * (max - min) + min

        INDArray min2 = toScale.min(0);
        INDArray max2 = toScale.max(0);

        INDArray std = toScale.subRowVector(min2).diviRowVector(max2.sub(min2));

        INDArray scaled = std.mul(max - min).addi(min);
        toScale.assign(scaled);
    }


}
