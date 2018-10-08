/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.util;

import lombok.val;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

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

        return timeSeriesMask.reshape('f', timeSeriesMask.length(), 1);
    }


    /**
     * Reshape time series mask arrays. This should match the assumptions (f order, etc) in RnnOutputLayer
     * @param timeSeriesMask    Mask array to reshape to a column vector
     * @return                  Mask array as a column vector
     */
    public static INDArray reshapeTimeSeriesMaskToVector(INDArray timeSeriesMask, LayerWorkspaceMgr workspaceMgr, ArrayType arrayType) {
        if (timeSeriesMask.rank() != 2)
            throw new IllegalArgumentException("Cannot reshape mask: rank is not 2");

        if (timeSeriesMask.ordering() != 'f' || !Shape.hasDefaultStridesForShape(timeSeriesMask))
            timeSeriesMask = workspaceMgr.dup(arrayType, timeSeriesMask, 'f');

        return workspaceMgr.leverageTo(arrayType, timeSeriesMask.reshape('f', timeSeriesMask.length(), 1));
    }

    /**
     * Reshape time series mask arrays. This should match the assumptions (f order, etc) in RnnOutputLayer
     * This reshapes from [X,1] to [X,1,1,1] suitable for per-example masking in CNNs
     * @param timeSeriesMask    Mask array to reshape for CNN per-example masking
     * @return                  Mask array as 4D CNN mask array: [X, 1, 1, 1]
     */
    public static INDArray reshapeTimeSeriesMaskToCnn4dMask(INDArray timeSeriesMask, LayerWorkspaceMgr workspaceMgr, ArrayType arrayType) {
        if (timeSeriesMask.rank() != 2)
            throw new IllegalArgumentException("Cannot reshape mask: rank is not 2");

        if (timeSeriesMask.ordering() != 'f' || !Shape.hasDefaultStridesForShape(timeSeriesMask))
            timeSeriesMask = workspaceMgr.dup(arrayType, timeSeriesMask, 'f');

        return workspaceMgr.leverageTo(arrayType, timeSeriesMask.reshape('f', timeSeriesMask.length(), 1, 1, 1));
    }

    /**
     * Reshape time series mask arrays. This should match the assumptions (f order, etc) in RnnOutputLayer
     * @param timeSeriesMaskAsVector    Mask array to reshape to a column vector
     * @return                  Mask array as a column vector
     */
    public static INDArray reshapeVectorToTimeSeriesMask(INDArray timeSeriesMaskAsVector, int minibatchSize) {
        if (!timeSeriesMaskAsVector.isVector())
            throw new IllegalArgumentException("Cannot reshape mask: expected vector");

        val timeSeriesLength = timeSeriesMaskAsVector.length() / minibatchSize;

        return timeSeriesMaskAsVector.reshape('f', minibatchSize, timeSeriesLength);
    }

    /**
     * Reshape CNN-style 4d mask array of shape [seqLength*minibatch,1,1,1] to time series mask [mb,seqLength]
     * This should match the assumptions (f order, etc) in RnnOutputLayer
     * @param timeSeriesMaskAsCnnMask    Mask array to reshape
     * @return                  Sequence mask array - [minibatch, sequenceLength]
     */
    public static INDArray reshapeCnnMaskToTimeSeriesMask(INDArray timeSeriesMaskAsCnnMask, int minibatchSize) {
        Preconditions.checkArgument(timeSeriesMaskAsCnnMask.rank() == 4 || timeSeriesMaskAsCnnMask.size(1) != 1 ||
                timeSeriesMaskAsCnnMask.size(2) != 1 || timeSeriesMaskAsCnnMask.size(3) != 1,
                "Expected rank 4 mask with shape [mb*seqLength, 1, 1, 1]. Got rank %s mask array with shape %s",
                timeSeriesMaskAsCnnMask.rank(), timeSeriesMaskAsCnnMask.shape());

        val timeSeriesLength = timeSeriesMaskAsCnnMask.length() / minibatchSize;

        return timeSeriesMaskAsCnnMask.reshape('f', minibatchSize, timeSeriesLength);
    }

    public static INDArray reshapePerOutputTimeSeriesMaskTo2d(INDArray perOutputTimeSeriesMask) {
        if (perOutputTimeSeriesMask.rank() != 3) {
            throw new IllegalArgumentException(
                    "Cannot reshape per output mask: rank is not 3 (is: " + perOutputTimeSeriesMask.rank()
                            + ", shape = " + Arrays.toString(perOutputTimeSeriesMask.shape()) + ")");
        }

        return reshape3dTo2d(perOutputTimeSeriesMask);
    }

    public static INDArray reshapePerOutputTimeSeriesMaskTo2d(INDArray perOutputTimeSeriesMask, LayerWorkspaceMgr workspaceMgr, ArrayType arrayType) {
        if (perOutputTimeSeriesMask.rank() != 3) {
            throw new IllegalArgumentException(
                            "Cannot reshape per output mask: rank is not 3 (is: " + perOutputTimeSeriesMask.rank()
                                            + ", shape = " + Arrays.toString(perOutputTimeSeriesMask.shape()) + ")");
        }

        return reshape3dTo2d(perOutputTimeSeriesMask, workspaceMgr, arrayType);
    }

    public static INDArray reshape3dTo2d(INDArray in) {
        if (in.rank() != 3)
            throw new IllegalArgumentException("Invalid input: expect NDArray with rank 3");
        val shape = in.shape();
        if (shape[0] == 1)
            return in.tensorAlongDimension(0, 1, 2).permutei(1, 0); //Edge case: miniBatchSize==1
        if (shape[2] == 1)
            return in.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        INDArray permuted = in.permute(0, 2, 1); //Permute, so we get correct order after reshaping
        return permuted.reshape('f', shape[0] * shape[2], shape[1]);
    }

    public static INDArray reshape3dTo2d(INDArray in, LayerWorkspaceMgr workspaceMgr, ArrayType arrayType) {
        if (in.rank() != 3)
            throw new IllegalArgumentException("Invalid input: expect NDArray with rank 3");
        val shape = in.shape();
        INDArray ret;
        if (shape[0] == 1) {
            ret = in.tensorAlongDimension(0, 1, 2).permutei(1, 0); //Edge case: miniBatchSize==1
        } else if (shape[2] == 1) {
            ret = in.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        } else {
            INDArray permuted = in.permute(0, 2, 1); //Permute, so we get correct order after reshaping
            ret = permuted.reshape('f', shape[0] * shape[2], shape[1]);
        }
        return workspaceMgr.leverageTo(arrayType, ret);
    }

    public static INDArray reshape2dTo3d(INDArray in, int miniBatchSize) {
        if (in.rank() != 2)
            throw new IllegalArgumentException("Invalid input: expect NDArray with rank 2");
        //Based on: RnnToFeedForwardPreProcessor
        val shape = in.shape();
        if (in.ordering() != 'f')
            in = Shape.toOffsetZeroCopy(in, 'f');
        INDArray reshaped = in.reshape('f', miniBatchSize, shape[0] / miniBatchSize, shape[1]);
        return reshaped.permute(0, 2, 1);
    }


    public static INDArray reshape2dTo3d(INDArray in, int miniBatchSize, LayerWorkspaceMgr workspaceMgr, ArrayType arrayType) {
        if (in.rank() != 2)
            throw new IllegalArgumentException("Invalid input: expect NDArray with rank 2");
        //Based on: RnnToFeedForwardPreProcessor
        val shape = in.shape();
        if (in.ordering() != 'f') {
            in = workspaceMgr.dup(arrayType, in, 'f');
        }
        INDArray reshaped = in.reshape('f', miniBatchSize, shape[0] / miniBatchSize, shape[1]);
        return workspaceMgr.leverageTo(arrayType, reshaped.permute(0, 2, 1));
    }

    /**
     * Reverse an input time series along the time dimension
     *
     * @param in Input activations to reverse, with shape [minibatch, size, timeSeriesLength]
     * @return Reversed activations
     */
    public static INDArray reverseTimeSeries(INDArray in){
        if(in == null){
            return null;
        }

        if(in.ordering() != 'f' || in.isView() || !Shape.strideDescendingCAscendingF(in)){
            in = in.dup('f');
        }

        // FIXME: int cast
        int[] idxs = new int[(int) in.size(2)];
        int j=0;
        for( int i=idxs.length-1; i>=0; i--){
            idxs[j++] = i;
        }

        INDArray inReshape = in.reshape('f', in.size(0)*in.size(1), in.size(2));

        INDArray outReshape = Nd4j.pullRows(inReshape, 0, idxs, 'f');
        return outReshape.reshape('f', in.size(0), in.size(1), in.size(2));
    }

    /**
     * Reverse an input time series along the time dimension
     *
     * @param in Input activations to reverse, with shape [minibatch, size, timeSeriesLength]
     * @return Reversed activations
     */
    public static INDArray reverseTimeSeries(INDArray in, LayerWorkspaceMgr workspaceMgr, ArrayType arrayType){
        if(in == null){
            return null;
        }

        if(in.ordering() != 'f' || in.isView() || !Shape.strideDescendingCAscendingF(in)){
            in = workspaceMgr.dup(arrayType, in, 'f');
        }

        // FIXME: int cast
        int[] idxs = new int[(int) in.size(2)];
        int j=0;
        for( int i=idxs.length-1; i>=0; i--){
            idxs[j++] = i;
        }

        INDArray inReshape = in.reshape('f', in.size(0)*in.size(1), in.size(2));

        INDArray outReshape = workspaceMgr.create(arrayType, new long[]{inReshape.size(0), idxs.length}, 'f');
        Nd4j.pullRows(inReshape, outReshape, 0, idxs);
        return workspaceMgr.leverageTo(arrayType, outReshape.reshape('f', in.size(0), in.size(1), in.size(2)));

        /*
        INDArray out = Nd4j.createUninitialized(in.shape(), 'f');
        CustomOp op = DynamicCustomOp.builder("reverse")
                .addIntegerArguments(new int[]{0,1})
                .addInputs(in)
                .addOutputs(out)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);
        return out;
        */
    }

    /**
     * Reverse a (per time step) time series mask, with shape [minibatch, timeSeriesLength]
     * @param mask Mask to reverse along time dimension
     * @return Mask after reversing
     */
    public static INDArray reverseTimeSeriesMask(INDArray mask){
        if(mask == null){
            return null;
        }
        if(mask.rank() == 3){
            //Should normally not be used - but handle the per-output masking case
            return reverseTimeSeries(mask);
        } else if(mask.rank() != 2){
            throw new IllegalArgumentException("Invalid mask rank: must be rank 2 or 3. Got rank " + mask.rank()
                    + " with shape " + Arrays.toString(mask.shape()));
        }

        // FIXME: int cast
        int[] idxs = new int[(int) mask.size(1)];
        int j=0;
        for( int i=idxs.length-1; i>=0; i--){
            idxs[j++] = i;
        }

        return Nd4j.pullRows(mask, 0, idxs);
    }


    /**
     * Reverse a (per time step) time series mask, with shape [minibatch, timeSeriesLength]
     * @param mask Mask to reverse along time dimension
     * @return Mask after reversing
     */
    public static INDArray reverseTimeSeriesMask(INDArray mask, LayerWorkspaceMgr workspaceMgr, ArrayType arrayType){
        if(mask == null){
            return null;
        }
        if(mask.rank() == 3){
            //Should normally not be used - but handle the per-output masking case
            return reverseTimeSeries(mask, workspaceMgr, arrayType);
        } else if(mask.rank() != 2){
            throw new IllegalArgumentException("Invalid mask rank: must be rank 2 or 3. Got rank " + mask.rank()
                    + " with shape " + Arrays.toString(mask.shape()));
        }

        // FIXME: int cast
        int[] idxs = new int[(int) mask.size(1)];
        int j=0;
        for( int i=idxs.length-1; i>=0; i--){
            idxs[j++] = i;
        }

        INDArray ret = workspaceMgr.createUninitialized(arrayType, new long[]{mask.size(0), idxs.length}, 'f');

        return Nd4j.pullRows(mask, ret, 0, idxs);

        /*
        //Assume input mask is 2d: [minibatch, tsLength]
        INDArray out = Nd4j.createUninitialized(mask.shape(), 'f');
        CustomOp op = DynamicCustomOp.builder("reverse")
                .addIntegerArguments(new int[]{1})
                .addInputs(mask)
                .addOutputs(out)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);
        return out;
        */
    }

    /**
     * Extract out the last time steps (2d array from 3d array input) accounting for the mask layer, if present.
     *
     * @param pullFrom Input time series array (rank 3) to pull the last time steps from
     * @param mask     Mask array (rank 2). May be null
     * @return         2d array of the last time steps
     */
    public static Pair<INDArray,int[]> pullLastTimeSteps(INDArray pullFrom, INDArray mask){
        //Then: work out, from the mask array, which time step of activations we want, extract activations
        //Also: record where they came from (so we can do errors later)
        int[] fwdPassTimeSteps;
        INDArray out;
        if (mask == null) {

            // FIXME: int cast
            //No mask array -> extract same (last) column for all
            int lastTS = (int) pullFrom.size(2) - 1;
            out = pullFrom.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(lastTS));
            fwdPassTimeSteps = null; //Null -> last time step for all examples
        } else {
            val outShape = new long[] {pullFrom.size(0), pullFrom.size(1)};
            out = Nd4j.create(outShape);

            //Want the index of the last non-zero entry in the mask array
            INDArray lastStepArr = BooleanIndexing.lastIndex(mask, Conditions.epsNotEquals(0.0), 1);
            fwdPassTimeSteps = lastStepArr.data().asInt();

            //Now, get and assign the corresponding subsets of 3d activations:
            for (int i = 0; i < fwdPassTimeSteps.length; i++) {
                //TODO can optimize using reshape + pullRows
                out.putRow(i, pullFrom.get(NDArrayIndex.point(i), NDArrayIndex.all(),
                        NDArrayIndex.point(fwdPassTimeSteps[i])));
            }
        }

        return new Pair<>(out, fwdPassTimeSteps);
    }

    /**
     * Extract out the last time steps (2d array from 3d array input) accounting for the mask layer, if present.
     *
     * @param pullFrom Input time series array (rank 3) to pull the last time steps from
     * @param mask     Mask array (rank 2). May be null
     * @return         2d array of the last time steps
     */
    public static Pair<INDArray,int[]> pullLastTimeSteps(INDArray pullFrom, INDArray mask, LayerWorkspaceMgr workspaceMgr, ArrayType arrayType){
        //Then: work out, from the mask array, which time step of activations we want, extract activations
        //Also: record where they came from (so we can do errors later)
        int[] fwdPassTimeSteps;
        INDArray out;
        if (mask == null) {

            // FIXME: int cast
            //No mask array -> extract same (last) column for all
            int lastTS = (int) pullFrom.size(2) - 1;
            out = pullFrom.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(lastTS));
            fwdPassTimeSteps = null; //Null -> last time step for all examples
        } else {
            val outShape = new long[] {pullFrom.size(0), pullFrom.size(1)};
            out = Nd4j.create(outShape);

            //Want the index of the last non-zero entry in the mask array
            INDArray lastStepArr = BooleanIndexing.lastIndex(mask, Conditions.epsNotEquals(0.0), 1);
            fwdPassTimeSteps = lastStepArr.data().asInt();

            //Now, get and assign the corresponding subsets of 3d activations:
            for (int i = 0; i < fwdPassTimeSteps.length; i++) {
                //TODO can optimize using reshape + pullRows
                out.putRow(i, pullFrom.get(NDArrayIndex.point(i), NDArrayIndex.all(),
                        NDArrayIndex.point(fwdPassTimeSteps[i])));
            }
        }

        return new Pair<>(workspaceMgr.leverageTo(arrayType, out), fwdPassTimeSteps);
    }
}
