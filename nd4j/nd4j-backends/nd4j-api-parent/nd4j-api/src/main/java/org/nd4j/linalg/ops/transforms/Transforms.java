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

package org.nd4j.linalg.ops.transforms;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.reduce3.*;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.transforms.floating.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.ELUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.OldAtan2Op;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.And;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Not;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Or;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Xor;
import org.nd4j.linalg.api.ops.impl.transforms.same.*;
import org.nd4j.linalg.api.ops.impl.transforms.strict.OldSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Stabilize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;

/**
 * Functional interface for the different op classes
 *
 * @author Adam Gibson
 */
public class Transforms {


    private Transforms() {}

    /**
     * Cosine similarity
     *
     * @param d1 the first vector
     * @param d2 the second vector
     * @return the cosine similarities between the 2 arrays
     *
     */
    public static double cosineSim(@NonNull INDArray d1, @NonNull INDArray d2) {
        return Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(d1, d2, d1.length())).getFinalResult()
                        .doubleValue();
    }

    public static double cosineDistance(@NonNull INDArray d1, @NonNull INDArray d2) {
        return Nd4j.getExecutioner().execAndReturn(new CosineDistance(d1, d2, d1.length())).getFinalResult()
                        .doubleValue();
    }

    public static double hammingDistance(@NonNull INDArray d1, @NonNull INDArray d2) {
        return Nd4j.getExecutioner().execAndReturn(new HammingDistance(d1, d2, d1.length())).getFinalResult()
                        .doubleValue();
    }

    public static double jaccardDistance(@NonNull INDArray d1, @NonNull INDArray d2) {
        return Nd4j.getExecutioner().execAndReturn(new JaccardDistance(d1, d2, d1.length())).getFinalResult()
                        .doubleValue();
    }

    public static INDArray allCosineSimilarities(@NonNull INDArray d1, @NonNull INDArray d2, int... dimensions) {
        return Nd4j.getExecutioner().exec(new CosineSimilarity(d1, d2, true), dimensions);
    }

    public static INDArray allCosineDistances(@NonNull INDArray d1, @NonNull INDArray d2, int... dimensions) {
        return Nd4j.getExecutioner().exec(new CosineDistance(d1, d2, true), dimensions);
    }

    public static INDArray allEuclideanDistances(@NonNull INDArray d1, @NonNull INDArray d2, int... dimensions) {
        return Nd4j.getExecutioner().exec(new EuclideanDistance(d1, d2, true), dimensions);
    }

    public static INDArray allManhattanDistances(@NonNull INDArray d1, @NonNull INDArray d2, int... dimensions) {
        return Nd4j.getExecutioner().exec(new ManhattanDistance(d1, d2, true), dimensions);
    }


    public static INDArray reverse(INDArray x, boolean dup) {
        return Nd4j.getExecutioner().exec(new OldReverse(x, dup ? Nd4j.createUninitialized(x.shape(), x.ordering()) : x))
                        .z();
    }

    /**
     *
     * @param d1
     * @param d2
     * @return
     */
    public static double manhattanDistance(@NonNull INDArray d1, @NonNull INDArray d2) {
        return d1.distance1(d2);
    }

    /**
     * Atan2 operation, new INDArray instance will be returned
     * Note the order of x and y parameters is opposite to that of java.lang.Math.atan2
     *
     * @param x the abscissa coordinate
     * @param y the ordinate coordinate
     * @return the theta from point (r, theta) when converting (x,y) from to cartesian to polar coordinates
     */
    public static INDArray atan2(@NonNull INDArray x, @NonNull INDArray y) {
        return Nd4j.getExecutioner()
                        .execAndReturn(new OldAtan2Op(x, y, Nd4j.createUninitialized(x.shape(), x.ordering())));
    }

    /**
     *
     * @param d1
     * @param d2
     * @return
     */
    public static double euclideanDistance(@NonNull INDArray d1, @NonNull INDArray d2) {
        return d1.distance2(d2);
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
            if (toScale.data().dataType() == (DataType.FLOAT))
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
        return neg(ndArray, true);
    }


    /**
     * Binary matrix of whether the number at a given index is greater than
     *
     * @param ndArray
     * @return
     */
    public static INDArray floor(INDArray ndArray) {
        return floor(ndArray, true);

    }

    /**
     * Binary matrix of whether the number at a given index is greater than
     *
     * @param ndArray
     * @return
     */
    public static INDArray ceiling(INDArray ndArray) {
        return ceiling(ndArray, true);

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
        return sign(toSign, true);
    }


    /**
     *
     * @param ndArray
     * @param k
     * @return
     */
    public static INDArray stabilize(INDArray ndArray, double k) {
        return stabilize(ndArray, k, true);
    }

    /**
     * Sin function
     * @param in
     * @return
     */
    public static INDArray sin(INDArray in) {
        return sin(in, true);
    }

    /**
     * Sin function
     * @param in
     * @param copy
     * @return
     */
    public static INDArray sin(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new Sin((copy ? in.dup() : in)));
    }


    /**
     * Sin function
     * @param in
     * @return
     */
    public static INDArray atanh(INDArray in) {
        return atanh(in, true);
    }

    /**
     * Sin function
     * @param in
     * @param copy
     * @return
     */
    public static INDArray atanh(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new ATanh((copy ? in.dup() : in)));
    }

    /**
     * Sinh function
     * @param in
     * @return
     */
    public static INDArray sinh(INDArray in) {
        return sinh(in, true);
    }

    /**
     * Sinh function
     * @param in
     * @param copy
     * @return
     */
    public static INDArray sinh(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new Sinh((copy ? in.dup() : in)));
    }

    /**
     *
     * @param in
     * @return
     */
    public static INDArray cos(INDArray in) {
        return cos(in, true);
    }

    /**
     *
     * @param in
     * @param copy
     * @return
     */
    public static INDArray cosh(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new Cosh((copy ? in.dup() : in)));
    }

    /**
     *
     * @param in
     * @return
     */
    public static INDArray cosh(INDArray in) {
        return cosh(in, true);
    }

    /**
     *
     * @param in
     * @param copy
     * @return
     */
    public static INDArray cos(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new Cos((copy ? in.dup() : in)));
    }


    public static INDArray acos(INDArray arr) {
        return acos(arr, true);
    }


    public static INDArray acos(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new ACos(((copy ? in.dup() : in))));
    }


    public static INDArray asin(INDArray arr) {
        return asin(arr, true);
    }


    public static INDArray asin(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new ASin(((copy ? in.dup() : in))));
    }

    public static INDArray atan(INDArray arr) {
        return atan(arr, true);
    }


    public static INDArray atan(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new ATan(((copy ? in.dup() : in))));
    }

    public static INDArray ceil(INDArray arr) {
        return ceil(arr, true);
    }


    public static INDArray ceil(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new Ceil(((copy ? in.dup() : in))));
    }


    public static INDArray relu(INDArray arr) {
        return relu(arr, true);
    }


    public static INDArray relu(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new RectifedLinear(((copy ? in.dup() : in))));
    }

    public static INDArray relu6(INDArray arr) {
        return relu6(arr, true);
    }


    public static INDArray relu6(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new Relu6(((copy ? in.dup() : in))));
    }


    public static INDArray leakyRelu(INDArray arr) {
        return leakyRelu(arr, true);
    }


    public static INDArray leakyRelu(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new LeakyReLU(((copy ? in.dup() : in))));
    }

    public static INDArray elu(INDArray arr) {
        return elu(arr, true);
    }


    public static INDArray elu(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new ELU(((copy ? in.dup() : in))));
    }

    public static INDArray eluDerivative(INDArray arr) {
        return eluDerivative(arr, true);
    }


    public static INDArray eluDerivative(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new ELUDerivative(((copy ? in.dup() : in))));
    }



    public static INDArray leakyRelu(INDArray arr, double cutoff) {
        return leakyRelu(arr, cutoff, true);
    }


    public static INDArray leakyRelu(INDArray in, double cutoff, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new LeakyReLU((copy ? in.dup() : in), cutoff));
    }

    public static INDArray leakyReluDerivative(INDArray arr, double cutoff) {
        return leakyReluDerivative(arr, cutoff, true);
    }


    public static INDArray leakyReluDerivative(INDArray in, double cutoff, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new LeakyReLUDerivative((copy ? in.dup() : in), cutoff));
    }



    public static INDArray softPlus(INDArray arr) {
        return softPlus(arr, true);
    }


    public static INDArray softPlus(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new SoftPlus(((copy ? in.dup() : in))));
    }

    public static INDArray step(INDArray arr) {
        return softPlus(arr, true);
    }


    public static INDArray step(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new Step(((copy ? in.dup() : in))));
    }


    public static INDArray softsign(INDArray arr) {
        return softPlus(arr, true);
    }


    public static INDArray softsign(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new SoftSign(((copy ? in.dup() : in))));
    }


    public static INDArray softsignDerivative(INDArray arr) {
        return softPlus(arr, true);
    }


    public static INDArray softsignDerivative(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new SoftSignDerivative(((copy ? in.dup() : in))));
    }



    public static INDArray softmax(INDArray arr) {
        return softmax(arr, true);
    }


    /**
     *
     * @param in
     * @param copy
     * @return
     */
    public static INDArray softmax(INDArray in, boolean copy) {
        return Nd4j.getExecutioner().execAndReturn(new OldSoftMax(((copy ? in.dup() : in))));
    }

    /**
     * Abs function
     *
     * @param ndArray
     * @return
     */
    public static INDArray abs(INDArray ndArray) {
        return abs(ndArray, true);
    }


    /**
     * Run the exp operation
     * @param ndArray
     * @return
     */
    public static INDArray exp(INDArray ndArray) {
        return exp(ndArray, true);
    }


    public static INDArray hardTanh(INDArray ndArray) {
        return hardTanh(ndArray, true);

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


    public static INDArray hardTanhDerivative(INDArray ndArray) {
        return hardTanhDerivative(ndArray, true);

    }

    /**
     * Hard tanh
     *
     * @param ndArray the input
     * @param dup     whether to duplicate the ndarray and return it as the result
     * @return the output
     */
    public static INDArray hardTanhDerivative(INDArray ndArray, boolean dup) {
        return exec(dup ? new HardTanhDerivative(ndArray, ndArray.dup()) : new HardTanhDerivative(ndArray));
    }



    /**
     *
     * @param ndArray
     * @return
     */
    public static INDArray identity(INDArray ndArray) {
        return identity(ndArray, true);
    }


    /**
     * Pow function
     *
     * @param ndArray the ndarray to raise hte power of
     * @param power   the power to raise by
     * @return the ndarray raised to this power
     */
    public static INDArray pow(INDArray ndArray, Number power) {
        return pow(ndArray, power, true);

    }


    /**
     * Element-wise power function - x^y, performed element-wise.
     * Not performed in-place: the input arrays are not modified.
     *
     * @param ndArray the ndarray to raise to the power of
     * @param power   the power to raise by
     * @return a copy of the ndarray raised to the specified power (element-wise)
     */
    public static INDArray pow(INDArray ndArray, INDArray power) {
        return pow(ndArray, power, true);
    }

    /**
     * Element-wise power function - x^y, performed element-wise
     *
     * @param ndArray the ndarray to raise to the power of
     * @param power   the power to raise by
     * @param dup     if true:
     * @return the ndarray raised to this power
     */
    public static INDArray pow(INDArray ndArray, INDArray power, boolean dup) {
        INDArray result = (dup ? Nd4j.create(ndArray.shape(), ndArray.ordering()) : ndArray);
        return exec(new Pow(ndArray, power, result, ndArray.length(), 0));
    }

    /**
     * Rounding function
     *
     * @param ndArray
     * @return
     */
    public static INDArray round(INDArray ndArray) {
        return round(ndArray, true);
    }

    /**
     * Sigmoid function
     *
     * @param ndArray
     * @return
     */
    public static INDArray sigmoid(INDArray ndArray) {
        return sigmoid(ndArray, true);
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
     * Sigmoid function
     *
     * @param ndArray
     * @return
     */
    public static INDArray sigmoidDerivative(INDArray ndArray) {
        return sigmoid(ndArray, true);
    }

    /**
     * Sigmoid function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray sigmoidDerivative(INDArray ndArray, boolean dup) {
        return exec(dup ? new SigmoidDerivative(ndArray, Nd4j.createUninitialized(ndArray.shape(), ndArray.ordering())) : new SigmoidDerivative(ndArray));
    }


    /**
     * Sqrt function
     *
     * @param ndArray
     * @return
     */
    public static INDArray sqrt(INDArray ndArray) {
        return sqrt(ndArray, true);
    }


    /**
     * Element-wise tan function. Copies the array
     *
     * @param ndArray Input array
     */
    public static INDArray tan(INDArray ndArray) {
        return tan(ndArray, true);
    }

    /**
     * Element-wise tan function. Copies the array
     *
     * @param ndArray Input array
     */
    public static INDArray tan(INDArray ndArray, boolean dup) {
        return exec(dup ? new Tan(ndArray, Nd4j.createUninitialized(ndArray.shape(), ndArray.ordering())) : new Tan(ndArray));
    }

    /**
     * Tanh function
     *
     * @param ndArray
     * @return
     */
    public static INDArray tanh(INDArray ndArray) {
        return tanh(ndArray, true);
    }

    /**
     * Log on arbitrary base
     *
     * @param ndArray
     * @param base
     * @return
     */
    public static INDArray log(INDArray ndArray, double base) {
        return log(ndArray, base, true);
    }

    /**
     * Log on arbitrary base
     *
     * @param ndArray
     * @param base
     * @return
     */
    public static INDArray log(INDArray ndArray, double base, boolean duplicate) {
        return Nd4j.getExecutioner().exec(new LogX(duplicate ? ndArray.dup(ndArray.ordering()) : ndArray, base)).z();
    }

    public static INDArray log(INDArray ndArray) {
        return log(ndArray, true);
    }

    public static INDArray eps(INDArray ndArray) {
        return exec(new Eps(ndArray));
    }

    /**
     * 1 if greater than or equal to 0 otherwise (at each element)
     *
     * @param first
     * @param ndArray
     * @return
     */
    public static INDArray greaterThanOrEqual(INDArray first, INDArray ndArray) {
        return greaterThanOrEqual(first, ndArray, true);
    }

    /**
     * 1 if less than or equal to 0 otherwise (at each element)
     *
     * @param first
     * @param ndArray
     * @return
     */
    public static INDArray lessThanOrEqual(INDArray first, INDArray ndArray) {
        return lessThanOrEqual(first, ndArray, true);
    }


    /**
     * Eps function
     *
     * @param ndArray
     * @return
     */
    public static INDArray lessThanOrEqual(INDArray first, INDArray ndArray, boolean dup) {
        return exec(dup ? new OldLessThanOrEqual(first, ndArray, Nd4j.createUninitialized(first.shape()), first.length()) : new OldLessThanOrEqual(first, ndArray, first, first.length()));

    }


    /**
     * Eps function
     *
     * @param ndArray
     * @return
     */
    public static INDArray greaterThanOrEqual(INDArray first, INDArray ndArray, boolean dup) {
        return exec(dup ? new OldGreaterThanOrEqual(first, ndArray, Nd4j.createUninitialized(first.shape()), first.length()) : new OldGreaterThanOrEqual(first, ndArray, first, first.length()));

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
     * Maximum function with a scalar
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
     * Maximum function with a scalar
     *
     * @param ndArray tbe ndarray
     * @param k
     * @return
     */
    public static INDArray max(INDArray ndArray, double k) {
        return max(ndArray, k, true);
    }

    /**
     * Element wise maximum function between 2 INDArrays
     *
     * @param first
     * @param second
     * @param dup
     * @return
     */
    public static INDArray max(INDArray first, INDArray second, boolean dup) {
        if (dup) {
            first = first.dup();
        }
        return exec(new OldMax(second, first, first, first.length()));
    }

    /**
     * Element wise maximum function between 2 INDArrays
     *
     * @param first
     * @param second
     * @return
     */
    public static INDArray max(INDArray first, INDArray second) {
        return max(first, second, true);
    }

    /**
     * Minimum function with a scalar
     *
     * @param ndArray tbe ndarray
     * @param k
     * @param dup
     * @return
     */
    public static INDArray min(INDArray ndArray, double k, boolean dup) {
        return exec(dup ? new ScalarMin(ndArray.dup(), k) : new ScalarMin(ndArray, k));
    }

    /**
     * Maximum function with a scalar
     *
     * @param ndArray tbe ndarray
     * @param k
     * @return
     */
    public static INDArray min(INDArray ndArray, double k) {
        return min(ndArray, k, true);
    }

    /**
     * Element wise minimum function between 2 INDArrays
     *
     * @param first
     * @param second
     * @param dup
     * @return
     */
    public static INDArray min(INDArray first, INDArray second, boolean dup) {
        if (dup) {
            first = first.dup();
        }
        return exec(new OldMin(second, first, first, first.length()));
    }

    /**
     * Element wise minimum function between 2 INDArrays
     *
     * @param first
     * @param second
     * @return
     */
    public static INDArray min(INDArray first, INDArray second) {
        return min(first, second, true);
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
     * Elementwise exponential - 1 function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray expm1(INDArray ndArray, boolean dup) {
        return exec(dup ? new Expm1(ndArray, ndArray.dup()) : new Expm1(ndArray));
    }


    /**
     * Identity function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray identity(INDArray ndArray, boolean dup) {
        return exec(dup ? new OldIdentity(ndArray, ndArray.dup()) : new OldIdentity(ndArray));
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
     * Log of x + 1 function
     *
     * @param ndArray
     * @param dup
     * @return
     */
    public static INDArray log1p(INDArray ndArray, boolean dup) {
        return exec(dup ? new Log1p(ndArray, ndArray.dup()) : new Log1p(ndArray));
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

    public static INDArray and(INDArray x, INDArray y) {
        INDArray z = Nd4j.createUninitialized(x.shape(), x.ordering());
        Nd4j.getExecutioner().exec(new And(x, y, z, 0.0));
        return z;
    }

    public static INDArray or(INDArray x, INDArray y) {
        INDArray z = Nd4j.createUninitialized(x.shape(), x.ordering());
        Nd4j.getExecutioner().exec(new Or(x, y, z, 0.0));
        return z;
    }

    public static INDArray xor(INDArray x, INDArray y) {
        INDArray z = Nd4j.createUninitialized(x.shape(), x.ordering());
        Nd4j.getExecutioner().exec(new Xor(x, y, z, 0.0));
        return z;
    }

    public static INDArray not(INDArray x) {
        INDArray z = Nd4j.createUninitialized(x.shape(), x.ordering());
        Nd4j.getExecutioner().exec(new Not(x, z, 0.0));
        return z;
    }



    /**
     * Apply the given elementwise op
     *
     * @param op the factory to create the op
     * @return the new ndarray
     */
    private static INDArray exec(ScalarOp op) {
        if (op.x().isCleanedUp())
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
        if (op.x().isCleanedUp())
            throw new IllegalStateException("NDArray already freed");
        return Nd4j.getExecutioner().execAndReturn(op);
    }

    /**
     * Raises a square matrix to a power <i>n</i>, which can be positive, negative, or zero.
     * The behavior is similar to the numpy matrix_power() function.  The algorithm uses
     * repeated squarings to minimize the number of mmul() operations needed
     * <p>If <i>n</i> is zero, the identity matrix is returned.</p>
     * <p>If <i>n</i> is negative, the matrix is inverted and raised to the abs(n) power.</p>
     * @param in A square matrix to raise to an integer power, which will be changed if dup is false.
     * @param n The integer power to raise the matrix to.
     * @param dup If dup is true, the original input is unchanged.
     * @return The result of raising <i>in</i> to the <i>n</i>th power.
     */
    public static INDArray mpow(INDArray in, int n, boolean dup) {
        assert in.rows() == in.columns();
        if (n == 0) {
            if (dup)
                return Nd4j.eye(in.rows());
            else
                return in.assign(Nd4j.eye(in.rows()));
        }
        INDArray temp;
        if (n < 0) {
            temp = InvertMatrix.invert(in, !dup);
            n = -n;
        } else
            temp = in.dup();
        INDArray result = temp.dup();
        if (n < 4) {
            for (int i = 1; i < n; i++) {
                result.mmuli(temp);
            }
            if (dup)
                return result;
            else
                return in.assign(result);
        } else {
            // lets try to optimize by squaring itself a bunch of times
            int squares = (int) (Math.log(n) / Math.log(2.0));
            for (int i = 0; i < squares; i++)
                result = result.mmul(result);
            int diff = (int) Math.round(n - Math.pow(2.0, squares));
            for (int i = 0; i < diff; i++)
                result.mmuli(temp);
            if (dup)
                return result;
            else
                return in.assign(result);
        }
    }

}
