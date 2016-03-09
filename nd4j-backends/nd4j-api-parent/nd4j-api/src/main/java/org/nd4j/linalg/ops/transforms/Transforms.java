/*
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
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Functional interface for the different op classes
 *
 * @author Adam Gibson
 */
public class Transforms {



    /**
     * Cosine similarity
     *
     * @param d1 the first vector
     * @param d2 the second vector
     * @return the cosine similarities between the 2 arrays
     *
     */
    public static double cosineSim(INDArray d1, INDArray d2) {
        return Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(d1, d2, d1.length())).getFinalResult().doubleValue();
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
        double length = toScale.norm2Number().doubleValue();

        if (length > 0) {
            if (toScale.data().dataType() == (DataBuffer.Type.FLOAT))
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
     * Binary matrix of whether the number at a given index is greater than
     *
     * @param ndArray
     * @return
     */
    public static INDArray ceiling(INDArray ndArray) {
        return ceiling(ndArray, Nd4j.copyOnOps);

    }

    /**
     * Ceiling function
     * @param ndArray
     * @param copyOnOps
     * @return
     */
    public static INDArray ceiling(INDArray ndArray, boolean copyOnOps) {
        return exec(copyOnOps ? new Ceil(ndArray, ndArray.dup()) : new Ceil(ndArray, ndArray));
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

    public static INDArray sin(INDArray in){
        return sin(in, Nd4j.copyOnOps);
    }

    public static INDArray sin(INDArray in, boolean copy){
        return Nd4j.getExecutioner().execAndReturn(new Sin((copy ? in.dup() : in)));
    }

    /**
     *
     * @param in
     * @return
     */
    public static INDArray cos(INDArray in) {
        return cos(in, Nd4j.copyOnOps);
    }

    /**
     *
     * @param in
     * @param copy
     * @return
     */
    public static INDArray cos(INDArray in, boolean copy){
        return Nd4j.getExecutioner().execAndReturn(new Cos((copy ? in.dup() : in)));
    }


    public static INDArray acos(INDArray arr) {
        return acos(arr,Nd4j.copyOnOps);
    }


   public static INDArray acos(INDArray in,boolean copy) {
       return Nd4j.getExecutioner().execAndReturn(new ACos(((copy ? in.dup() : in))));
   }


    public static INDArray asin(INDArray arr) {
        return asin(arr, Nd4j.copyOnOps);
    }


    public static INDArray asin(INDArray in,boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new ASin(((copy ? in.dup() : in))));
    }

    public static INDArray atan(INDArray arr) {
        return acos(arr,Nd4j.copyOnOps);
    }


    public static INDArray atan(INDArray in,boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new ATan(((copy ? in.dup() : in))));
    }

    public static INDArray ceil(INDArray arr) {
        return ceil(arr, Nd4j.copyOnOps);
    }


    public static INDArray ceil(INDArray in,boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new Ceil(((copy ? in.dup() : in))));
    }


    public static INDArray relu(INDArray arr) {
        return relu(arr, Nd4j.copyOnOps);
    }


    public static INDArray relu(INDArray in,boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new RectifedLinear(((copy ? in.dup() : in))));
    }



    public static INDArray leakyRelu(INDArray arr) {
        return leakyRelu(arr, Nd4j.copyOnOps);
    }


    public static INDArray leakyRelu(INDArray in,boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new LeakyReLU(((copy ? in.dup() : in))));
    }


    public static INDArray softPlus(INDArray arr) {
        return softPlus(arr,Nd4j.copyOnOps);
    }


    public static INDArray softPlus(INDArray in,boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new SoftPlus(((copy ? in.dup() : in))));
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


    /**
     *
     * @param ndArray
     * @return
     */
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
