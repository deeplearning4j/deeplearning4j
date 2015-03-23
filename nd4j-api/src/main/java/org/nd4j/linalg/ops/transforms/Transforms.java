/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMax;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.GreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.LessThanOrEqual;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;


/**
 * Functional interface for the different op classes
 *
 * @author Adam Gibson
 */
public class Transforms {


    /**
     * Max pooling
     *
     * @param input
     * @param ds    the strides with which to pool expectations
     * @return
     * @parma ignoreBorder whether to ignore the borders of images
     */
    public static INDArray maxPool(INDArray input, int[] ds, boolean ignoreBorder) {
        assert input.length() >= 2 : "Max pooling requires an ndarray of >= length 2";
        assert ds.length == 2 : "Down sampling must be of length 2 (the factors used for each image size";
        assert input.shape().length == 4 : "Only supports 4 dimensional tensors";
        int batchSize = ArrayUtil.prod(new int[]{input.size(0) * input.size(1)});
        //possibly look at a input implementation instead (looping over the outer dimension slice wise with calling input repeatedly)
        //use the given rows and columns if ignoring borders
        int rows = input.size(2);
        int cols = input.size(3);

        INDArray signalNDArray = input.reshape(batchSize, 1, rows, cols);
        INDArray zz = Nd4j.create(signalNDArray.shape()).assign(Float.MIN_VALUE);

        int rowIter = ignoreBorder ? rows / (int) Math.pow(ds[0], 2) : rows;
        int colIter = ignoreBorder ? cols / (int) Math.pow(ds[1], 2) : cols;

        for (int i = 0; i < signalNDArray.size(0); i++) {
            for (int j = 0; j < signalNDArray.size(1); j++) {
                for (int k = 0; k < rowIter; k++) {
                    int zk = k / ds[0];
                    for (int l = 0; l < colIter; l++) {
                        int zl = l / ds[1];
                        double num = input.getDouble(i, j, k, l);
                        double zzGet = zz.getDouble(i, j, zk, zl);
                        zz.putScalar(new int[]{i, j, zk, zl}, Math.max(num, zzGet));
                    }
                }
            }
        }

        return zz.reshape(signalNDArray.shape());
    }

    /**
     * Down sampling a signal (specifically the first 2 dimensions)
     *
     * @param d1
     * @param stride
     * @return
     */
    public static INDArray downSample(INDArray d1, int[] stride) {
        INDArray d = Nd4j.ones(stride);
        d.divi(ArrayUtil.prod(stride));
        INDArray ret = Convolution.convn(d1, d, Convolution.Type.VALID);
        ret = ret.get(NDArrayIndex.interval(0, stride[0]), NDArrayIndex.interval(0, stride[1]));
        return ret;
    }


    /**
     * Pooled expectations
     *
     * @param toPool the ndarray to pool
     * @param stride the 2d stride across the ndarray
     * @return
     */
    public static INDArray pool(INDArray toPool, int[] stride) {

        int nDims = toPool.shape().length;
        assert nDims == 3 : "NDArray must have 3 dimensions";
        int nRows = toPool.shape()[nDims - 2];
        int nCols = toPool.shape()[nDims - 1];
        int yStride = stride[0], xStride = stride[1];
        INDArray blocks = Nd4j.create(toPool.shape());
        for (int iR = 0; iR < Math.ceil(nRows / yStride); iR++) {
            NDArrayIndex rows = NDArrayIndex.interval(iR * yStride, iR * yStride, true);
            for (int jC = 0; jC < Math.ceil(nCols / xStride); jC++) {
                NDArrayIndex cols = NDArrayIndex.interval(jC * xStride, (jC * xStride) + 1, true);
                INDArray blockVal = toPool.get(rows, cols).sum(toPool.shape().length - 1).sum(toPool.shape().length - 1);
                blocks.put(
                        new NDArrayIndex[]{rows, cols},
                        blockVal.permute(new int[]{1, 2, 0}))
                        .repmat(new int[]{rows.length(), cols.length()});
            }
        }

        return blocks;
    }

    /**
     * Upsampling a signal (specifically the first 2 dimensions
     *
     * @param d
     * @param scale
     * @return
     */
    public static INDArray upSample(INDArray d, INDArray scale) {

        INDArray idx = Nd4j.create(d.shape().length, 1);


        for (int i = 0; i < d.shape().length; i++) {
            INDArray tmp = Nd4j.zeros(d.size(i) * (int) scale.getDouble(i), 1);
            int[] indices = ArrayUtil.range(0, (int) scale.getDouble(i) * d.size(i), (int) scale.getDouble(i));
            tmp.putScalar(indices, 1.0f);
            idx.put(i,
                    tmp.cumsum(Integer.MAX_VALUE).sum(Integer.MAX_VALUE));
        }
        return idx;
    }


    /**
     * Cosine similarity
     *
     * @param d1
     * @param d2
     * @return
     */
    public static double cosineSim(INDArray d1, INDArray d2) {
        return Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(d1, d2, d1.length())).currentResult().doubleValue();
    }

    /**
     * Normalize data to zero mean and unit variance
     * substract by the mean and divide by the standard deviation
     *
     * @param toNormalize the ndarray to normalize
     * @return the normalized ndarray
     */
    public static INDArray normalizeZeroMeanAndUnitVariance(INDArray toNormalize) {
        INDArray columnMeans = toNormalize.mean(0);
        INDArray columnStds = toNormalize.std(0);

        toNormalize.subiRowVector(columnMeans);
        //padding for non zero
        columnStds.addi(Nd4j.EPS_THRESHOLD);
        toNormalize.diviRowVector(columnStds);
        return toNormalize;
    }


    /**
     * Scale by 1 / norm2 of the matrix
     *
     * @param toScale the ndarray to scale
     * @return the scaled ndarray
     */
    public static INDArray unitVec(INDArray toScale) {
        double length = toScale.norm2(Integer.MAX_VALUE).getDouble(0);

        if (length > 0) {
            if (toScale.data().dataType() == (DataBuffer.FLOAT))
                return Nd4j.getBlasWrapper().scal(1.0f / (float) length, toScale);
            else
                return Nd4j.getBlasWrapper().scal(1.0 / length, toScale);

        }
        return toScale;
    }


    /**
     * Returns the negative of an ndarray
     *
     * @param ndArray the ndarray to take the negative of
     * @return the negative of the ndarray
     */
    public static INDArray neg(INDArray ndArray) {
        return neg(ndArray, Nd4j.copyOnOps);
    }


    /**
     * Binary matrix of whether the number at a given index is greater than
     *
     * @param ndArray
     * @return
     */
    public static INDArray floor(INDArray ndArray) {
        return floor(ndArray, Nd4j.copyOnOps);

    }


    /**
     * Signum function of this ndarray
     *
     * @param toSign
     * @return
     */
    public static INDArray sign(INDArray toSign) {
        return sign(toSign, Nd4j.copyOnOps);
    }


    public static INDArray stabilize(INDArray ndArray, double k) {
        return stabilize(ndArray, k, Nd4j.copyOnOps);
    }


    /**
     * Abs funciton
     *
     * @param ndArray
     * @return
     */
    public static INDArray abs(INDArray ndArray) {
        return abs(ndArray, true);
    }


    public static INDArray exp(INDArray ndArray) {
        return exp(ndArray, Nd4j.copyOnOps);
    }


    public static INDArray hardTanh(INDArray ndArray) {
        return hardTanh(ndArray, Nd4j.copyOnOps);

    }


    public static INDArray identity(INDArray ndArray) {
        return identity(ndArray, Nd4j.copyOnOps);
    }


    /**
     * Pow function
     *
     * @param ndArray the ndarray to raise hte power of
     * @param power   the power to raise by
     * @return the ndarray raised to this power
     */
    public static INDArray pow(INDArray ndArray, Number power) {
        return pow(ndArray, power, Nd4j.copyOnOps);

    }

    /**
     * Rounding function
     *
     * @param ndArray
     * @return
     */
    public static INDArray round(INDArray ndArray) {
        return round(ndArray, Nd4j.copyOnOps);
    }

    /**
     * Sigmoid function
     *
     * @param ndArray
     * @return
     */
    public static INDArray sigmoid(INDArray ndArray) {
        return sigmoid(ndArray, Nd4j.copyOnOps);
    }

    /**
     * Sqrt function
     *
     * @param ndArray
     * @return
     */
    public static INDArray sqrt(INDArray ndArray) {
        return sqrt(ndArray, Nd4j.copyOnOps);
    }

    /**
     * Tanh function
     *
     * @param ndArray
     * @return
     */
    public static INDArray tanh(INDArray ndArray) {
        return tanh(ndArray, Nd4j.copyOnOps);
    }


    public static INDArray log(INDArray ndArray) {
        return log(ndArray, Nd4j.copyOnOps);
    }

    public static INDArray eps(INDArray ndArray) {
        return eps(ndArray, Nd4j.copyOnOps);
    }

    /**
     * 1 if greater than or equal to 0 otherwise (at each element)
     *
     * @param first
     * @param ndArray
     * @return
     */
    public static INDArray greaterThanOrEqual(INDArray first, INDArray ndArray) {
        return greaterThanOrEqual(first, ndArray, Nd4j.copyOnOps);
    }

    /**
     * 1 if less than or equal to 0 otherwise (at each element)
     *
     * @param first
     * @param ndArray
     * @return
     */
    public static INDArray lessThanOrEqual(INDArray first, INDArray ndArray) {
        return lessThanOrEqual(first, ndArray, Nd4j.copyOnOps);
    }


    /**
     * Eps function
     *
     * @param ndArray
     * @return
     */
    public static INDArray lessThanOrEqual(INDArray first, INDArray ndArray, boolean dup) {
        return exec(dup ? new LessThanOrEqual(first.dup(), ndArray) : new LessThanOrEqual(first, ndArray));

    }


    /**
     * Eps function
     *
     * @param ndArray
     * @return
     */
    public static INDArray greaterThanOrEqual(INDArray first, INDArray ndArray, boolean dup) {
        return exec(dup ? new GreaterThanOrEqual(first.dup(), ndArray) : new GreaterThanOrEqual(first, ndArray));

    }


    /**
     * Eps function
     *
     * @param ndArray
     * @return
     */
    public static INDArray eps(INDArray ndArray, boolean dup) {
        return exec(dup ? new Eps(ndArray.dup()) : new Eps(ndArray));

    }


    /**
     * Floor function
     *
     * @param ndArray
     * @return
     */
    public static INDArray floor(INDArray ndArray, boolean dup) {
        return exec(dup ? new Floor(ndArray.dup()) : new Floor(ndArray));

    }


    /**
     * Signum function of this ndarray
     *
     * @param toSign
     * @return
     */
    public static INDArray sign(INDArray toSign, boolean dup) {
        return exec(dup ? new Sign(toSign, toSign.dup()) : new Sign(toSign));
    }

    /**
     * Stabilize to be within a range of k
     *
     * @param ndArray tbe ndarray
     * @param k
     * @param dup
     * @return
     */
    public static INDArray max(INDArray ndArray, double k, boolean dup) {
        return exec(dup ? new ScalarMax(ndArray.dup(), k) : new ScalarMax(ndArray, k));
    }

    /**
     * Stabilize to be within a range of k
     *
     * @param ndArray tbe ndarray
     * @param k
     * @return
     */
    public static INDArray max(INDArray ndArray, double k) {
        return max(ndArray, k, Nd4j.copyOnOps);
    }


    /**
     * Stabilize to be within a range of k
     *
     * @param ndArray tbe ndarray
     * @param k
     * @param dup
     * @return
     */
    public static INDArray stabilize(INDArray ndArray, double k, boolean dup) {
        return exec(dup ? new Stabilize(ndArray, ndArray.dup(), k) : new Stabilize(ndArray, k));
    }


    /**
     * Abs function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray abs(INDArray ndArray, boolean dup) {
        return exec(dup ? new Abs(ndArray, ndArray.dup()) : new Abs(ndArray));

    }

    /**
     * Exp function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray exp(INDArray ndArray, boolean dup) {
        return exec(dup ? new Exp(ndArray, ndArray.dup()) : new Exp(ndArray));
    }


    /**
     * Hard tanh
     *
     * @param ndArray the input
     * @param dup     whether to duplicate the ndarray and return it as the result
     * @return the output
     */
    public static INDArray hardTanh(INDArray ndArray, boolean dup) {
        return exec(dup ? new HardTanh(ndArray, ndArray.dup()) : new HardTanh(ndArray));
    }

    /**
     * Identity function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray identity(INDArray ndArray, boolean dup) {
        return exec(dup ? new Identity(ndArray, ndArray.dup()) : new Identity(ndArray));
    }


    /**
     * Pow function
     *
     * @param ndArray
     * @param power
     * @param dup
     * @return
     */
    public static INDArray pow(INDArray ndArray, Number power, boolean dup) {
        return exec(dup ? new Pow(ndArray, ndArray.dup(), power.doubleValue()) : new Pow(ndArray, power.doubleValue()));
    }

    /**
     * Rounding function
     *
     * @param ndArray the ndarray
     * @param dup
     * @return
     */
    public static INDArray round(INDArray ndArray, boolean dup) {
        return exec(dup ? new Round(ndArray, ndArray.dup()) : new Round(ndArray));
    }


    /**
     * Sigmoid function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray sigmoid(INDArray ndArray, boolean dup) {
        return exec(dup ? new Sigmoid(ndArray, ndArray.dup()) : new Sigmoid(ndArray));
    }

    /**
     * Sqrt function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray sqrt(INDArray ndArray, boolean dup) {
        return exec(dup ? new Sqrt(ndArray, ndArray.dup()) : new Sqrt(ndArray));
    }

    /**
     * Tanh function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray tanh(INDArray ndArray, boolean dup) {
        return exec(dup ? new Tanh(ndArray, ndArray.dup()) : new Tanh(ndArray));
    }

    /**
     * Log function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray log(INDArray ndArray, boolean dup) {
        return exec(dup ? new Log(ndArray, ndArray.dup()) : new Log(ndArray));
    }


    /**
     * Negative
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray neg(INDArray ndArray, boolean dup) {
        return exec(dup ? new Negative(ndArray, ndArray.dup()) : new Negative(ndArray));
    }

    /**
     * Apply the given elementwise op
     *
     * @param op the factory to create the op
     * @return the new ndarray
     */
    private static INDArray exec(ScalarOp op) {
        if(op.x().isCleanedUp())
            throw new IllegalStateException("NDArray already freed");
        return Nd4j.getExecutioner().exec(op).z();
    }

    /**
     * Apply the given elementwise op
     *
     * @param op the factory to create the op
     * @return the new ndarray
     */
    private static INDArray exec(TransformOp op) {
        if(op.x().isCleanedUp())
            throw new IllegalStateException("NDArray already freed");
        return Nd4j.getExecutioner().execAndReturn(op);
    }


}
