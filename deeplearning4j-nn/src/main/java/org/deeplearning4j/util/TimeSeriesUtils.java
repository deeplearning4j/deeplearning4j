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
 */

package org.deeplearning4j.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/**
 * Basic time series utils
 * @author Adam Gibson
 */
public class TimeSeriesUtils {


    private TimeSeriesUtils() {}

    /**
     * Calculate a moving average given the length
     * @param toAvg the array to average
     * @param n the length of the moving window
     * @return the moving averages for each row
     */
    public static INDArray movingAverage(INDArray toAvg, int n) {
        INDArray ret = Nd4j.cumsum(toAvg);
        INDArrayIndex[] ends = new INDArrayIndex[] {NDArrayIndex.interval(n, toAvg.columns())};
        INDArrayIndex[] begins = new INDArrayIndex[] {NDArrayIndex.interval(0, toAvg.columns() - n, false)};
        INDArrayIndex[] nMinusOne = new INDArrayIndex[] {NDArrayIndex.interval(n - 1, toAvg.columns())};
        ret.put(ends, ret.get(ends).sub(ret.get(begins)));
        return ret.get(nMinusOne).divi(n);
    }

    /**
     * Reshape time series mask arrays. This should match the assumptions (f order, etc) in RnnOutputLayer
     * @param timeSeriesMask    Mask array to reshape to a column vector
     * @return                  Mask array as a column vector
     */
    public static INDArray reshapeTimeSeriesMaskToVector(INDArray timeSeriesMask) {
        if (timeSeriesMask.rank() != 2)
            throw new IllegalArgumentException("Cannot reshape mask: rank is not 2");

        if (timeSeriesMask.ordering() != 'f')
            timeSeriesMask = timeSeriesMask.dup('f');

        return timeSeriesMask.reshape('f', new int[] {timeSeriesMask.length(), 1});
    }


    /**
     * Reshape time series mask arrays. This should match the assumptions (f order, etc) in RnnOutputLayer
     * @param timeSeriesMaskAsVector    Mask array to reshape to a column vector
     * @return                  Mask array as a column vector
     */
    public static INDArray reshapeVectorToTimeSeriesMask(INDArray timeSeriesMaskAsVector, int minibatchSize) {
        if (!timeSeriesMaskAsVector.isVector())
            throw new IllegalArgumentException("Cannot reshape mask: expected vector");

        int timeSeriesLength = timeSeriesMaskAsVector.length() / minibatchSize;

        return timeSeriesMaskAsVector.reshape('f', new int[] {minibatchSize, timeSeriesLength});
    }

    public static INDArray reshapePerOutputTimeSeriesMaskTo2d(INDArray perOutputTimeSeriesMask) {
        if (perOutputTimeSeriesMask.rank() != 3) {
            throw new IllegalArgumentException(
                            "Cannot reshape per output mask: rank is not 3 (is: " + perOutputTimeSeriesMask.rank()
                                            + ", shape = " + Arrays.toString(perOutputTimeSeriesMask.shape()) + ")");
        }

        return reshape3dTo2d(perOutputTimeSeriesMask);
    }

    public static INDArray reshape3dTo2d(INDArray in) {
        if (in.rank() != 3)
            throw new IllegalArgumentException("Invalid input: expect NDArray with rank 3");
        int[] shape = in.shape();
        if (shape[0] == 1)
            return in.tensorAlongDimension(0, 1, 2).permutei(1, 0); //Edge case: miniBatchSize==1
        if (shape[2] == 1)
            return in.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        INDArray permuted = in.permute(0, 2, 1); //Permute, so we get correct order after reshaping
        return permuted.reshape('f', shape[0] * shape[2], shape[1]);
    }

    public static INDArray reshape2dTo3d(INDArray in, int miniBatchSize) {
        if (in.rank() != 2)
            throw new IllegalArgumentException("Invalid input: expect NDArray with rank 2");
        //Based on: RnnToFeedForwardPreProcessor
        int[] shape = in.shape();
        if (in.ordering() != 'f')
            in = Shape.toOffsetZeroCopy(in, 'f');
        INDArray reshaped = in.reshape('f', miniBatchSize, shape[0] / miniBatchSize, shape[1]);
        return reshaped.permute(0, 2, 1);
    }

}
