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

package org.nd4j.autodiff.functions;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.Data;
import lombok.NonNull;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.NoOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd;
import org.nd4j.linalg.api.ops.impl.broadcast.BiasAddGrad;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Enter;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Exit;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.NextIteration;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Switch;
import org.nd4j.linalg.api.ops.impl.image.ExtractImagePatches;
import org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMin;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.layers.convolution.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;
import org.nd4j.linalg.api.ops.impl.loss.AbsoluteDifferenceLoss;
import org.nd4j.linalg.api.ops.impl.loss.CosineDistanceLoss;
import org.nd4j.linalg.api.ops.impl.loss.HingeLoss;
import org.nd4j.linalg.api.ops.impl.loss.HuberLoss;
import org.nd4j.linalg.api.ops.impl.loss.L2Loss;
import org.nd4j.linalg.api.ops.impl.loss.LogLoss;
import org.nd4j.linalg.api.ops.impl.loss.LogPoissonLoss;
import org.nd4j.linalg.api.ops.impl.loss.MeanPairwiseSquaredErrorLoss;
import org.nd4j.linalg.api.ops.impl.loss.MeanSquaredErrorLoss;
import org.nd4j.linalg.api.ops.impl.loss.SigmoidCrossEntropyLoss;
import org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss;
import org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyWithLogitsLoss;
import org.nd4j.linalg.api.ops.impl.loss.SparseSoftmaxCrossEntropyLossWithLogits;
import org.nd4j.linalg.api.ops.impl.loss.WeightedCrossEntropyLoss;
import org.nd4j.linalg.api.ops.impl.loss.bp.AbsoluteDifferenceLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.CosineDistanceLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.HingeLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.HuberLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.LogLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.LogPoissonLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.MeanPairwiseSquaredErrorLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.MeanSquaredErrorLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.SigmoidCrossEntropyLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.SoftmaxCrossEntropyLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.SoftmaxCrossEntropyWithLogitsLossBp;
import org.nd4j.linalg.api.ops.impl.loss.bp.SparseSoftmaxCrossEntropyLossWithLogitsBp;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.reduce.MmulBp;
import org.nd4j.linalg.api.ops.impl.reduce.Moments;
import org.nd4j.linalg.api.ops.impl.reduce.NormalizeMoments;
import org.nd4j.linalg.api.ops.impl.reduce.TensorMmul;
import org.nd4j.linalg.api.ops.impl.reduce.ZeroFraction;
import org.nd4j.linalg.api.ops.impl.reduce.bool.All;
import org.nd4j.linalg.api.ops.impl.reduce.bool.Any;
import org.nd4j.linalg.api.ops.impl.reduce.bp.CumProdBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.CumSumBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.DotBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.MaxBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.MeanBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.MinBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.Norm1Bp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.Norm2Bp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.NormMaxBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.ProdBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.SquaredNormBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.StandardDeviationBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.SumBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.VarianceBp;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BatchMmul;
import org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp;
import org.nd4j.linalg.api.ops.impl.reduce.floating.AMean;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Entropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.LogEntropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Mean;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax;
import org.nd4j.linalg.api.ops.impl.reduce.floating.ShannonEntropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountZero;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.impl.reduce.same.AMax;
import org.nd4j.linalg.api.ops.impl.reduce.same.AMin;
import org.nd4j.linalg.api.ops.impl.reduce.same.ASum;
import org.nd4j.linalg.api.ops.impl.reduce.same.Max;
import org.nd4j.linalg.api.ops.impl.reduce.same.Min;
import org.nd4j.linalg.api.ops.impl.reduce.same.Prod;
import org.nd4j.linalg.api.ops.impl.reduce.same.Sum;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.reduce3.Dot;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.HammingDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.JaccardDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThanOrEqual;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarNotEquals;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterAdd;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterDiv;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterMax;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterMin;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterMul;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterSub;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix;
import org.nd4j.linalg.api.ops.impl.shape.Cross;
import org.nd4j.linalg.api.ops.impl.shape.Diag;
import org.nd4j.linalg.api.ops.impl.shape.DiagPart;
import org.nd4j.linalg.api.ops.impl.shape.ExpandDims;
import org.nd4j.linalg.api.ops.impl.shape.Gather;
import org.nd4j.linalg.api.ops.impl.shape.GatherNd;
import org.nd4j.linalg.api.ops.impl.shape.MergeAvg;
import org.nd4j.linalg.api.ops.impl.shape.MergeMax;
import org.nd4j.linalg.api.ops.impl.shape.MeshGrid;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.api.ops.impl.shape.OnesLike;
import org.nd4j.linalg.api.ops.impl.shape.ParallelStack;
import org.nd4j.linalg.api.ops.impl.shape.Permute;
import org.nd4j.linalg.api.ops.impl.shape.Rank;
import org.nd4j.linalg.api.ops.impl.shape.ReductionShape;
import org.nd4j.linalg.api.ops.impl.shape.Repeat;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.api.ops.impl.shape.SequenceMask;
import org.nd4j.linalg.api.ops.impl.shape.Size;
import org.nd4j.linalg.api.ops.impl.shape.SizeAt;
import org.nd4j.linalg.api.ops.impl.shape.Slice;
import org.nd4j.linalg.api.ops.impl.shape.Squeeze;
import org.nd4j.linalg.api.ops.impl.shape.Stack;
import org.nd4j.linalg.api.ops.impl.shape.StridedSlice;
import org.nd4j.linalg.api.ops.impl.shape.Tile;
import org.nd4j.linalg.api.ops.impl.shape.Transpose;
import org.nd4j.linalg.api.ops.impl.shape.Unstack;
import org.nd4j.linalg.api.ops.impl.shape.ZerosLike;
import org.nd4j.linalg.api.ops.impl.shape.bp.SliceBp;
import org.nd4j.linalg.api.ops.impl.shape.bp.StridedSliceBp;
import org.nd4j.linalg.api.ops.impl.shape.bp.TileBp;
import org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.Constant;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;
import org.nd4j.linalg.api.ops.impl.transforms.ReluLayer;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.bool.IsFinite;
import org.nd4j.linalg.api.ops.impl.transforms.bool.IsInf;
import org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN;
import org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.api.ops.impl.transforms.custom.*;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Pow;
import org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMax;
import org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMean;
import org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMin;
import org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentProd;
import org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentSum;
import org.nd4j.linalg.api.ops.impl.transforms.dtype.Cast;
import org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.CubeDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.DynamicPartitionBp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.ELUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.LogSoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.RationalTanhDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.RectifiedTanhDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.Relu6Derivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SELUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.*;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.*;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.And;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Or;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Xor;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.same.Ceil;
import org.nd4j.linalg.api.ops.impl.transforms.same.Cube;
import org.nd4j.linalg.api.ops.impl.transforms.same.Floor;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;
import org.nd4j.linalg.api.ops.impl.transforms.same.Negative;
import org.nd4j.linalg.api.ops.impl.transforms.same.Reciprocal;
import org.nd4j.linalg.api.ops.impl.transforms.same.Round;
import org.nd4j.linalg.api.ops.impl.transforms.same.Sign;
import org.nd4j.linalg.api.ops.impl.transforms.same.Square;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMax;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMean;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMin;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentProd;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSqrtN;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSum;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentMaxBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentMeanBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentMinBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentProdBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.SegmentSumBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentMaxBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentMeanBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentMinBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentProdBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentSqrtNBp;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.UnsortedSegmentSumBp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ACos;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ASin;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ASinh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ATan;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ATanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Cos;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Cosh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erf;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erfc;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Expm1;
import org.nd4j.linalg.api.ops.impl.transforms.strict.GELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.GELUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.HardSigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.HardTanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log1p;
import org.nd4j.linalg.api.ops.impl.transforms.strict.LogSigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.RationalTanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.RectifiedTanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sin;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sinh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SoftPlus;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SoftSign;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Swish;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SwishDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tan;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh;
import org.nd4j.linalg.api.ops.random.custom.DistributionUniform;
import org.nd4j.linalg.api.ops.random.custom.RandomBernoulli;
import org.nd4j.linalg.api.ops.random.custom.RandomExponential;
import org.nd4j.linalg.api.ops.random.custom.RandomNormal;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.api.ops.random.impl.BinomialDistribution;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.api.ops.random.impl.LogNormalDistribution;
import org.nd4j.linalg.api.ops.random.impl.Range;
import org.nd4j.linalg.api.ops.random.impl.TruncatedNormalDistribution;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.util.ArrayUtil;

/**
 *
 */
@Data
public class DifferentialFunctionFactory {

    protected SameDiff sameDiff;
    private static Map<String, Method> methodNames;

    /**
     * @param sameDiff
     */
    public DifferentialFunctionFactory(SameDiff sameDiff) {
        if (sameDiff != null) {
            this.sameDiff = sameDiff;
            if (methodNames == null) {
                methodNames = new HashMap<>();
                Method[] methods = getClass().getDeclaredMethods();
                for (Method method : methods)
                    methodNames.put(method.getName().toLowerCase(), method);
            }
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }


    }

    public SameDiff sameDiff() {
        return sameDiff;
    }


    public SDVariable invoke(String name, Object[] args) {
        try {
            return (SDVariable) methodNames.get(name).invoke(this, args);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    public Constant val(SDVariable iX) {
        return new Constant(sameDiff(), iX,
                iX.getShape());
    }

    public ExternalErrorsFunction externalErrors(SDVariable... inputs) {
        return externalErrors(null, inputs);
    }

    public ExternalErrorsFunction externalErrors(Map<String, INDArray> externalGradients, SDVariable... inputs) {
        Preconditions.checkArgument(inputs != null && inputs.length > 0, "Require at least one SDVariable to" +
                " be specified when using external errors: got %s", inputs);
        ExternalErrorsFunction fn = new ExternalErrorsFunction(sameDiff(), Arrays.asList(inputs), externalGradients);
        fn.outputVariable();
        return fn;
    }

    public SDVariable zerosLike(SDVariable input) {
        return zerosLike(null, input);
    }

    public SDVariable zerosLike(String name, SDVariable input) {
        validateDifferentialFunctionsameDiff(input);
        return new ZerosLike(name, sameDiff(), input).outputVariable();
    }

    public SDVariable onesLike(String name, SDVariable input, DataType dataType) {
        validateDifferentialFunctionsameDiff(input);
        return new OnesLike(name, sameDiff(), input, dataType).outputVariable();
    }

    public SDVariable constant(SDVariable input, long... shape) {
        return new Constant(sameDiff(), input, (shape != null && shape.length > 0 ? shape : null)).outputVariable();
    }

    public SDVariable linspace(SDVariable lower, SDVariable upper, SDVariable count, DataType dt) {
        return new org.nd4j.linalg.api.ops.impl.shape.Linspace(sameDiff(), lower, upper, count, dt).outputVariable();
    }

    public SDVariable range(double from, double to, double step, DataType dataType) {
        return new Range(sameDiff(), from, to, step, dataType).outputVariable();
    }

    public SDVariable range(SDVariable from, SDVariable to, SDVariable step, DataType dataType) {
        return new Range(sameDiff(), from, to, step, dataType).outputVariable();
    }

    public SDVariable[] listdiff(SDVariable x, SDVariable y){
        return new ListDiff(sameDiff(), x, y).outputVariables();
    }

    public SDVariable cast(SDVariable toCast, DataType toType){
        return new Cast(sameDiff(), toCast, toType).outputVariable();
    }

    public SDVariable[] meshgrid(boolean cartesian, SDVariable... inputs) {
        return new MeshGrid(sameDiff(), cartesian, inputs).outputVariables();
    }

    public SDVariable randomUniform(double min, double max, SDVariable shape) {
        return new DistributionUniform(sameDiff(), shape, min, max).outputVariable();
    }

    public SDVariable randomUniform(double min, double max, long... shape) {
        return new UniformDistribution(sameDiff(), min, max, shape).outputVariable();
    }

    public SDVariable randomNormal(double mean, double std, SDVariable shape) {
        return new RandomNormal(sameDiff(), shape, mean, std).outputVariable();
    }

    public SDVariable randomNormal(double mean, double std, long... shape) {
        return new GaussianDistribution(sameDiff(), mean, std, shape).outputVariable();
    }

    public SDVariable randomBernoulli(double p, SDVariable shape) {
        return new RandomBernoulli(sameDiff(), shape, p).outputVariable();
    }

    public SDVariable randomBernoulli(double p, long... shape) {
        return new BernoulliDistribution(sameDiff(), p, shape).outputVariable();
    }

    public SDVariable randomBinomial(int nTrials, double p, long... shape) {
        return new BinomialDistribution(sameDiff(), nTrials, p, shape).outputVariable();
    }

    public SDVariable randomLogNormal(double mean, double stdev, long... shape) {
        return new LogNormalDistribution(sameDiff(), mean, stdev, shape).outputVariable();
    }

    public SDVariable randomNormalTruncated(double mean, double stdev, long... shape) {
        return new TruncatedNormalDistribution(sameDiff(), mean, stdev, shape).outputVariable();
    }

    /**
     * Exponential distribution: P(x) = lambda * exp(-lambda * x)
     *
     * @param lambda Must be > 0
     * @param shape  Shape of the output
     */
    public SDVariable randomExponential(double lambda, SDVariable shape) {
        return new RandomExponential(sameDiff(), shape, lambda).outputVariable();
    }


    public SDVariable pad(SDVariable input, SDVariable padding, Pad.Mode mode, double padValue){
        return new Pad(sameDiff(), input, padding, mode, padValue).outputVariable();
    }

    /**
     * Local response normalization operation.
     *
     * @param input     the inputs to lrn
     * @param lrnConfig the configuration
     * @return
     */
    public SDVariable localResponseNormalization(SDVariable input, LocalResponseNormalizationConfig lrnConfig) {
        LocalResponseNormalization lrn = LocalResponseNormalization.builder()
                .inputFunctions(new SDVariable[]{input})
                .sameDiff(sameDiff())
                .config(lrnConfig)
                .build();

        return lrn.outputVariable();
    }

    /**
     * Conv1d operation.
     *
     * @param input        the inputs to conv1d
     * @param weights      conv1d weights
     * @param conv1DConfig the configuration
     * @return
     */
    public SDVariable conv1d(SDVariable input, SDVariable weights, Conv1DConfig conv1DConfig) {
        Conv1D conv1D = Conv1D.builder()
                .inputFunctions(new SDVariable[]{input, weights})
                .sameDiff(sameDiff())
                .config(conv1DConfig)
                .build();

        return conv1D.outputVariable();
    }

    /**
     * Conv2d operation.
     *
     * @param inputs       the inputs to conv2d
     * @param conv2DConfig the configuration
     * @return
     */
    public SDVariable conv2d(SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        Conv2D conv2D = Conv2D.builder()
                .inputFunctions(inputs)
                .sameDiff(sameDiff())
                .config(conv2DConfig)
                .build();

        return conv2D.outputVariable();
    }

    public SDVariable upsampling2d(SDVariable input, boolean nchw, int scaleH, int scaleW) {
        return new Upsampling2d(sameDiff(), input, nchw, scaleH, scaleW).outputVariable();
    }

    public SDVariable upsampling2dBp(SDVariable input, SDVariable gradient, boolean nchw, int scaleH, int scaleW) {
        return new Upsampling2dDerivative(sameDiff(), input, gradient, nchw, scaleH, scaleW).outputVariable();
    }


    /**
     * Average pooling 2d operation.
     *
     * @param input           the inputs to pooling
     * @param pooling2DConfig the configuration
     * @return
     */
    public SDVariable avgPooling2d(SDVariable input, Pooling2DConfig pooling2DConfig) {
        AvgPooling2D avgPooling2D = AvgPooling2D.builder()
                .input(input)
                .sameDiff(sameDiff())
                .config(pooling2DConfig)
                .build();

        return avgPooling2D.outputVariable();
    }

    /**
     * Max pooling 2d operation.
     *
     * @param input           the inputs to pooling
     * @param pooling2DConfig the configuration
     * @return
     */
    public SDVariable maxPooling2d(SDVariable input, Pooling2DConfig pooling2DConfig) {
        MaxPooling2D maxPooling2D = MaxPooling2D.builder()
                .input(input)
                .sameDiff(sameDiff())
                .config(pooling2DConfig)
                .build();

        return maxPooling2D.outputVariable();
    }

    /**
     * Avg pooling 3d operation.
     *
     * @param input           the inputs to pooling
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable avgPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        pooling3DConfig.setType(Pooling3D.Pooling3DType.AVG);
        return new AvgPooling3D(sameDiff(), input, pooling3DConfig).outputVariable();
    }


    /**
     * Max pooling 3d operation.
     *
     * @param input           the inputs to pooling
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable maxPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        pooling3DConfig.setType(Pooling3D.Pooling3DType.MAX);
        return new MaxPooling3D(sameDiff(), input, pooling3DConfig).outputVariable();
    }


    /**
     * Separable Conv2d operation.
     *
     * @param inputs       the inputs to conv2d
     * @param conv2DConfig the configuration
     * @return
     */
    public SDVariable sconv2d(SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        SConv2D sconv2D = SConv2D.sBuilder()
                .inputFunctions(inputs)
                .sameDiff(sameDiff())
                .conv2DConfig(conv2DConfig)
                .build();

        return sconv2D.outputVariable();
    }


    /**
     * Depth-wise Conv2d operation. This is just separable convolution with
     * only the depth-wise weights specified.
     *
     * @param inputs            the inputs to conv2d
     * @param depthConv2DConfig the configuration
     * @return
     */
    public SDVariable depthWiseConv2d(SDVariable[] inputs, Conv2DConfig depthConv2DConfig) {
        SConv2D depthWiseConv2D = SConv2D.sBuilder()
                .inputFunctions(inputs)
                .sameDiff(sameDiff())
                .conv2DConfig(depthConv2DConfig)
                .build();

        return depthWiseConv2D.outputVariable();
    }


    /**
     * Deconv2d operation.
     *
     * @param inputs         the inputs to conv2d
     * @param deconv2DConfig the configuration
     * @return
     */
    public SDVariable deconv2d(SDVariable[] inputs, DeConv2DConfig deconv2DConfig) {
        DeConv2D deconv2D = DeConv2D.builder()
                .inputs(inputs)
                .sameDiff(sameDiff())
                .config(deconv2DConfig)
                .build();

        return deconv2D.outputVariable();
    }

    public SDVariable deconv3d(SDVariable input, SDVariable weights, SDVariable bias, DeConv3DConfig config) {
        DeConv3D d = new DeConv3D(sameDiff(), input, weights, bias, config);
        return d.outputVariable();
    }

    public SDVariable[] deconv3dDerivative(SDVariable input, SDVariable weights, SDVariable bias, SDVariable grad, DeConv3DConfig config) {
        DeConv3DDerivative d = new DeConv3DDerivative(sameDiff(), input, weights, bias, grad, config);
        return d.outputVariables();
    }

    /**
     * Conv3d operation.
     *
     * @param inputs       the inputs to conv3d
     * @param conv3DConfig the configuration
     * @return
     */
    public SDVariable conv3d(SDVariable[] inputs, Conv3DConfig conv3DConfig) {
        Conv3D conv3D = Conv3D.builder()
                .inputFunctions(inputs)
                .conv3DConfig(conv3DConfig)
                .sameDiff(sameDiff())
                .build();

        val outputVars = conv3D.outputVariables();
        return outputVars[0];
    }


    /**
     * Batch norm operation.
     */
    public SDVariable batchNorm(SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta,
                                boolean applyGamma, boolean applyBeta,
                                double epsilon, int... axis) {
        BatchNorm batchNorm = BatchNorm.builder()
                .inputFunctions(new SDVariable[]{input, mean, variance, gamma, beta})
                .applyGamma(applyGamma)
                .applyBeta(applyBeta)
                .epsilon(epsilon)
                .sameDiff(sameDiff())
                .axis(axis)
                .build();

        val outputVars = batchNorm.outputVariables();
        return outputVars[0];
    }

    public SDVariable im2Col(SDVariable input, Conv2DConfig config) {
        return new Im2col(sameDiff(), input, config).outputVariable();
    }

    public SDVariable im2ColBp(SDVariable im2colInput, SDVariable gradientAtOutput, Conv2DConfig config) {
        return new Im2colBp(sameDiff(), im2colInput, gradientAtOutput, config).outputVariable();
    }

    public SDVariable col2Im(SDVariable input, Conv2DConfig config) {
        return new Col2Im(sameDiff(), input, config).outputVariable();
    }

    public SDVariable extractImagePatches(SDVariable input, int kH, int kW, int sH, int sW, int rH, int rW, boolean sameMode){
        return new ExtractImagePatches(sameDiff(), input, new int[]{kH, kW}, new int[]{sH, sW}, new int[]{rH, rW}, sameMode).outputVariable();
    }

    public SDVariable[] moments(SDVariable input, int... axes) {
        return new Moments(sameDiff(), input, axes).outputVariables();
    }

    public SDVariable[] normalizeMoments(SDVariable counts, SDVariable means, SDVariable variances, double shift) {
        return new NormalizeMoments(sameDiff(), counts, means, variances, shift).outputVariables();
    }


    public SDVariable tile(@NonNull SDVariable iX, @NonNull int[] repeat) {
        return new Tile(sameDiff(), iX, repeat).outputVariable();
    }

    public SDVariable tileBp(@NonNull SDVariable in, @NonNull SDVariable grad, @NonNull int[] repeat){
        return new TileBp(sameDiff, in, grad, repeat).outputVariable();
    }

    public SDVariable tile(@NonNull SDVariable iX, @NonNull SDVariable repeat) {
        return new Tile(sameDiff(), iX, repeat).outputVariable();
    }

    public SDVariable tileBp(@NonNull SDVariable in, @NonNull SDVariable repeat,  @NonNull SDVariable grad){
        return new TileBp(sameDiff, in, repeat, grad).outputVariable();
    }

    public SDVariable dropout(SDVariable input, double p) {
        return new DropOutInverted(sameDiff(), input, p).outputVariable();
    }


    public SDVariable sum(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new Sum(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable sumBp(SDVariable i_x, SDVariable grad, boolean keepDims, int... dimensions) {
        return new SumBp(sameDiff(), i_x, grad, keepDims, dimensions).outputVariable();
    }


    public SDVariable prod(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new Prod(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable prodBp(SDVariable preReduceInput, SDVariable grad, boolean keepDims, int... dimensions) {
        return new ProdBp(sameDiff(), preReduceInput, grad, keepDims, dimensions).outputVariable();
    }

    public SDVariable mean(SDVariable in, boolean keepDims, int... dimensions) {
        return new Mean(sameDiff(), in, keepDims, dimensions).outputVariable();
    }

    public SDVariable meanBp(SDVariable in, SDVariable grad, boolean keepDims, int... dimensions) {
        return new MeanBp(sameDiff(), in, grad, keepDims, dimensions).outputVariable();
    }


    public SDVariable std(SDVariable i_x, boolean biasCorrected, boolean keepDims, int... dimensions) {
        return new StandardDeviation(sameDiff(), i_x, biasCorrected, keepDims, dimensions).outputVariable();
    }

    public SDVariable stdBp(SDVariable stdInput, SDVariable gradient, boolean biasCorrected, boolean keepDims, int... dimensions) {
        return new StandardDeviationBp(sameDiff(), stdInput, gradient, biasCorrected, keepDims, dimensions).outputVariable();
    }


    public SDVariable variance(SDVariable i_x, boolean biasCorrected, boolean keepDims, int... dimensions) {
        return new Variance(sameDiff(), i_x, biasCorrected, keepDims, dimensions).outputVariable();
    }

    public SDVariable varianceBp(SDVariable stdInput, SDVariable gradient, boolean biasCorrected, boolean keepDims, int... dimensions) {
        return new VarianceBp(sameDiff(), stdInput, gradient, biasCorrected, keepDims, dimensions).outputVariable();
    }

    public SDVariable standardize(SDVariable i_x, int... dimensions) {
        return new Standardize(sameDiff(), i_x, dimensions).outputVariable();
    }

    public SDVariable standardizeBp(SDVariable stdInput, SDVariable gradient, int... dimensions) {
        return new StandardizeBp(sameDiff(), stdInput, gradient, dimensions).outputVariable();
    }

    public SDVariable layerNorm(SDVariable input, SDVariable gain, SDVariable bias, boolean channelsFirst, int... dimensions) {
        return new LayerNorm(sameDiff(), input, gain, bias, channelsFirst, dimensions).outputVariable();
    }

    public SDVariable[] layerNormBp(SDVariable input, SDVariable gain, SDVariable bias, SDVariable gradient, boolean channelsFirst, int... dimensions) {
        return new LayerNormBp(sameDiff(), input, gain, bias, gradient, channelsFirst, dimensions).outputVariables();
    }

    public SDVariable layerNorm(SDVariable input, SDVariable gain, boolean channelsFirst, int... dimensions) {
        return new LayerNorm(sameDiff(), input, gain, channelsFirst, dimensions).outputVariable();
    }

    public SDVariable[] layerNormBp(SDVariable input, SDVariable gain, SDVariable gradient, boolean channelsFirst, int... dimensions) {
        return new LayerNormBp(sameDiff(), input, gain, gradient, channelsFirst, dimensions).outputVariables();
    }

    public SDVariable squaredNorm(SDVariable input, boolean keepDims, int... dimensions) {
        return new SquaredNorm(sameDiff(), input, keepDims, dimensions).outputVariable();
    }

    public SDVariable squaredNormBp(SDVariable preReduceInput, SDVariable gradient, boolean keepDims, int... dimensions) {
        return new SquaredNormBp(sameDiff(), preReduceInput, gradient, keepDims, dimensions).outputVariable();
    }

    public SDVariable entropy(SDVariable in, int... dimensions) {
        return new Entropy(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable logEntropy(SDVariable in, int... dimensions) {
        return new LogEntropy(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable shannonEntropy(SDVariable in, int... dimensions){
        return new ShannonEntropy(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable countNonZero(SDVariable input, int... dimensions) {
        return new CountNonZero(sameDiff(), input, dimensions).outputVariable();
    }

    public SDVariable countZero(SDVariable input, int... dimensions) {
        return new CountZero(sameDiff(), input, dimensions).outputVariable();
    }

    public SDVariable zeroFraction(SDVariable input) {
        return new ZeroFraction(sameDiff(), input).outputVariable();
    }

    public SDVariable scalarMax(SDVariable in, Number num) {
        return new ScalarMax(sameDiff(), in, num).outputVariable();
    }

    public SDVariable scalarMin(SDVariable in, Number num) {
        return new ScalarMin(sameDiff(), in, num).outputVariable();
    }

    public SDVariable scalarSet(SDVariable in, Number num) {
        return new ScalarSet(sameDiff(), in, num).outputVariable();
    }

    public SDVariable scalarFloorMod(SDVariable in, Number num) {
        return new ScalarFMod(sameDiff(), in, num).outputVariable();
    }

    public SDVariable max(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new Max(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable max(SDVariable first, SDVariable second) {
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.Max(sameDiff(), first, second)
                .outputVariable();
    }

    public SDVariable maxBp(SDVariable i_x, SDVariable grad, boolean keepDims, int... dimensions) {
        return new MaxBp(sameDiff(), i_x, grad, keepDims, dimensions).outputVariable();
    }


    public SDVariable min(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new Min(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable minBp(SDVariable i_x, SDVariable grad, boolean keepDims, int... dimensions) {
        return new MinBp(sameDiff(), i_x, grad, keepDims, dimensions).outputVariable();
    }

    public SDVariable min(SDVariable first, SDVariable second) {
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.Min(sameDiff(), first, second)
                .outputVariable();
    }

    public SDVariable amax(SDVariable in, int... dimensions) {
        return new AMax(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable amin(SDVariable in, int... dimensions) {
        return new AMin(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable amean(SDVariable in, int... dimensions) {
        return new AMean(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable asum(SDVariable in, int... dimensions) {
        return new ASum(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable argmax(SDVariable in, boolean keepDims, int... dimensions) {
        return new IMax(sameDiff(), in, keepDims, dimensions).outputVariable();
    }

    public SDVariable argmin(SDVariable in, boolean keepDims, int... dimensions) {
        return new IMin(sameDiff(), in, keepDims, dimensions).outputVariable();
    }

    public SDVariable iamax(SDVariable in, boolean keepDims, int... dimensions) {
        return new IAMax(sameDiff(), in, keepDims, dimensions).outputVariable();
    }

    public SDVariable iamin(SDVariable in, boolean keepDims, int... dimensions) {
        return new IAMin(sameDiff(), in, keepDims, dimensions).outputVariable();
    }

    public SDVariable firstIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions) {
        return new FirstIndex(sameDiff(), in, condition, keepDims, dimensions).outputVariable();
    }

    public SDVariable lastIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions) {
        return new LastIndex(sameDiff(), in, condition, keepDims, dimensions).outputVariable();
    }

    /**
     * Returns a count of the number of elements that satisfy the condition
     *
     * @param in        Input
     * @param condition Condition
     * @return Number of elements that the condition is satisfied for
     */
    public SDVariable matchConditionCount(SDVariable in, Condition condition, boolean keepDims, int... dimensions) {
        return new MatchCondition(sameDiff(), in, condition, keepDims, dimensions).outputVariable();
    }

    /**
     * Returns a boolean mask of equal shape to the input, where the condition is satisfied
     *
     * @param in        Input
     * @param condition Condition
     * @return Boolean mask
     */
    public SDVariable matchCondition(SDVariable in, Condition condition) {
        return new MatchConditionTransform(sameDiff(), in, condition).outputVariable();
    }

    public SDVariable cumsum(SDVariable in, boolean exclusive, boolean reverse, int... axis) {
        return new CumSum(sameDiff(), in, exclusive, reverse, axis).outputVariable();
    }

    public SDVariable cumsumBp(SDVariable in, SDVariable grad, boolean exclusive, boolean reverse, int... axis) {
        return new CumSumBp(sameDiff(), in, grad, exclusive, reverse, axis).outputVariable();
    }

    public SDVariable cumprod(SDVariable in, boolean exclusive, boolean reverse, int... axis) {
        return new CumProd(sameDiff(), in, exclusive, reverse, axis).outputVariable();
    }

    public SDVariable cumprodBp(SDVariable in, SDVariable grad, boolean exclusive, boolean reverse, int... axis) {
        return new CumProdBp(sameDiff(), in, grad, exclusive, reverse, axis).outputVariable();
    }

    public SDVariable biasAdd(SDVariable input, SDVariable bias) {
        return new BiasAdd(sameDiff(), input, bias).outputVariable();
    }

    public SDVariable[] biasAddBp(SDVariable input, SDVariable bias, SDVariable grad) {
        return new BiasAddGrad(sameDiff(), input, bias, grad).outputVariables();
    }

    public SDVariable norm1(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new Norm1(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable norm1Bp(SDVariable preReduceIn, SDVariable grad, boolean keepDims, int... dimensions) {
        return new Norm1Bp(sameDiff(), preReduceIn, grad, keepDims, dimensions).outputVariable();
    }

    public SDVariable norm2(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new Norm2(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable norm2Bp(SDVariable preReduceIn, SDVariable grad, boolean keepDims, int... dimensions) {
        return new Norm2Bp(sameDiff(), preReduceIn, grad, keepDims, dimensions).outputVariable();
    }

    public SDVariable normmax(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new NormMax(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable normmaxBp(SDVariable preReduceIn, SDVariable grad, boolean keepDims, int... dimensions) {
        return new NormMaxBp(sameDiff(), preReduceIn, grad, keepDims, dimensions).outputVariable();
    }

    public SDVariable reductionShape(SDVariable shape, SDVariable axis, boolean keepDim){
        return new ReductionShape(sameDiff(), shape, axis, keepDim).outputVariable();
    }

    /**
     * Add 1s as required to the array make an array possible to be broadcast with the original (pre-reduce) array.
     * <p>
     * Example: if doing [a,b,c].sum(1), result is [a,c]. To 'undo' this in a way that can be auto-broadcast,
     * we want to expand as required - i.e., [a,c] -> [a,1,c] which can be auto-broadcast with the original [a,b,c].
     * This is typically only used with reduction operations backprop.
     *
     * @param origRank   Rank of the original array, before the reduction was executed
     * @param reduceDims Dimensions that the original array was reduced from
     * @param toExpand   Array to add 1s to the shape to (such that it can be
     * @return Reshaped array.
     */
    public SDVariable reductionBroadcastableWithOrigShape(int origRank, int[] reduceDims, SDVariable toExpand) {
        if (Shape.isWholeArray(origRank, reduceDims)) {
            //Output is [1,1] which is already broadcastable
            return toExpand;
        } else if (origRank == 2 && reduceDims.length == 1) {
            //In this case: [a,b] -> [1,b] or [a,b] -> [a,1]
            //both are already broadcastable
            return toExpand;
        } else {
            //Example: [a,b,c].sum(1) -> [a,c]... want [a,1,c]
            for (int d : reduceDims) {
                toExpand = sameDiff().expandDims(toExpand, d);
            }
            return toExpand;
        }
    }

    public SDVariable reductionBroadcastableWithOrigShape(SDVariable origInput, SDVariable axis, SDVariable toExpand) {
        SDVariable shape = origInput.shape();
        SDVariable reduceShape = reductionShape(shape, axis, true);
        SDVariable reshaped = toExpand.reshape(reduceShape);
        return reshaped;
    }


    public SDVariable gradientBackwardsMarker(SDVariable iX) {
        return new GradientBackwardsMarker(sameDiff(), iX, sameDiff.scalar(iX.getVarName() + "-pairgrad", 1.0)).outputVariable();
    }

    public SDVariable abs(SDVariable iX) {
        return new Abs(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable neg(SDVariable iX) {
        return new Negative(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable cos(SDVariable iX) {
        return new Cos(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable sin(SDVariable iX) {
        return new Sin(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable tan(SDVariable iX) {
        return new Tan(sameDiff(), iX, false).outputVariable();

    }


    public SDVariable permute(SDVariable iX, int... dimensions) {
        return new Permute(sameDiff(), iX, dimensions).outputVariable();
    }

    public SDVariable permute(SDVariable in, SDVariable dimensions) {
        return new Permute(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable noop(SDVariable input) {
        return new NoOp(sameDiff(), input).outputVariable();
    }

    public SDVariable identity(SDVariable input) {
        return new Identity(sameDiff(), input).outputVariable();
    }

    public SDVariable all(SDVariable input, int... dimensions) {
        return new All(sameDiff(), input, dimensions).outputVariable();
    }

    public SDVariable any(SDVariable input, int... dimensions) {
        return new Any(sameDiff(), input, dimensions).outputVariable();
    }

    public SDVariable invertPermutation(SDVariable input, boolean inPlace) {
        return new InvertPermutation(sameDiff(), input, inPlace).outputVariable();
    }

    public SDVariable transpose(SDVariable iX) {
        return new Transpose(sameDiff(), iX).outputVariable();
    }


    public SDVariable acos(SDVariable iX) {
        return new ACos(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable asin(SDVariable iX) {
        return new ASin(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable atan(SDVariable iX) {
        return new ATan(sameDiff(), iX, false).outputVariable();

    }

    public SDVariable atan2(SDVariable y, SDVariable x) {
        return new ATan2(sameDiff(), y, x).outputVariable();
    }


    public SDVariable cosh(SDVariable iX) {
        return new Cosh(sameDiff(), iX, false).outputVariable();

    }


    public SDVariable sinh(SDVariable iX) {
        return new Sinh(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable tanh(SDVariable iX) {
        return new Tanh(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable tanhRational(SDVariable in) {
        return new RationalTanh(sameDiff(), in, false).outputVariable();
    }

    public SDVariable tanhRectified(SDVariable in) {
        return new RectifiedTanh(sameDiff(), in, false).outputVariable();
    }

    public SDVariable tanhDerivative(SDVariable iX, SDVariable wrt) {
        return new org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative(sameDiff(), iX, wrt).outputVariable();
    }

    public SDVariable tanhRationalDerivative(SDVariable in) {
        return new RationalTanhDerivative(sameDiff(), in, false).outputVariable();
    }

    public SDVariable tanhRectifiedDerivative(SDVariable in) {
        return new RectifiedTanhDerivative(sameDiff(), in, false).outputVariable();
    }

    public SDVariable step(SDVariable in, double cutoff) {
        return new Step(sameDiff(), in, false, cutoff).outputVariable();
    }


    public SDVariable acosh(SDVariable iX) {
        return new ACosh(sameDiff(), iX).outputVariable();
    }


    public SDVariable asinh(SDVariable iX) {
        return new ASinh(sameDiff(), iX).outputVariable();
    }


    public SDVariable atanh(SDVariable iX) {
        return new ATanh(sameDiff(), iX).outputVariable();
    }


    public SDVariable exp(SDVariable iX) {
        return new Exp(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable expm1(SDVariable iX) {
        return new Expm1(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable rsqrt(SDVariable iX) {
        return new RSqrt(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable log(SDVariable iX) {
        return new Log(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable log(SDVariable in, double base) {
        return new LogX(sameDiff(), in, base).outputVariable();
    }

    public SDVariable log1p(SDVariable iX) {
        return new Log1p(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable isFinite(SDVariable ix) {
        return new IsFinite(sameDiff(), ix, false).outputVariable();
    }

    public SDVariable isInfinite(SDVariable ix) {
        return new IsInf(sameDiff(), ix, false).outputVariable();
    }

    public SDVariable isNaN(SDVariable ix) {
        return new IsNaN(sameDiff(), ix, false).outputVariable();
    }

    public SDVariable isMax(SDVariable ix) {
        return new IsMax(sameDiff(), ix).outputVariable();
    }

    public SDVariable replaceWhere(SDVariable to, SDVariable from, Condition condition) {
        return new CompareAndReplace(sameDiff(), to, from, condition).outputVariable();
    }

    public SDVariable replaceWhere(SDVariable to, Number set, Condition condition) {
        return new CompareAndSet(sameDiff(), to, set, condition).outputVariable();
    }

    public SDVariable round(SDVariable ix) {
        return new Round(sameDiff(), ix, false).outputVariable();
    }

    public SDVariable or(SDVariable iX, SDVariable i_y) {
        return new Or(sameDiff(), iX, i_y).outputVariable();
    }

    public SDVariable and(SDVariable ix, SDVariable iy) {
        return new And(sameDiff(), ix, iy).outputVariable();
    }

    public SDVariable xor(SDVariable ix, SDVariable iy) {
        return new Xor(sameDiff(), ix, iy).outputVariable();
    }

    public SDVariable shift(SDVariable ix, SDVariable shift) {
        return new ShiftBits(sameDiff(), ix, shift).outputVariable();
    }

    public SDVariable rshift(SDVariable ix, SDVariable shift) {
        return new RShiftBits(sameDiff(), ix, shift).outputVariable();
    }

    public SDVariable rotl(SDVariable ix, SDVariable shift) {
        return new CyclicShiftBits(sameDiff(), ix, shift).outputVariable();
    }

    public SDVariable rotr(SDVariable ix, SDVariable shift) {
        return new CyclicRShiftBits(sameDiff(), ix, shift).outputVariable();
    }

    public SDVariable eq(SDVariable iX, SDVariable i_y) {
        return new EqualTo(sameDiff(), new SDVariable[]{iX, i_y}, false).outputVariable();
    }


    public SDVariable neq(SDVariable iX, double i_y) {
        return new ScalarNotEquals(sameDiff(), iX, i_y).outputVariable();
    }


    public SDVariable neqi(SDVariable iX, double i_y) {
        return new ScalarNotEquals(sameDiff(), iX, i_y, true).outputVariable();
    }


    public SDVariable neqi(SDVariable iX, SDVariable i_y) {
        return new NotEqualTo(sameDiff(), new SDVariable[]{iX, i_y}, true).outputVariable();
    }

    public SDVariable neq(SDVariable iX, SDVariable i_y) {
        return new NotEqualTo(sameDiff(), new SDVariable[]{iX, i_y}, false).outputVariable();
    }

    public SDVariable pow(SDVariable iX, double i_y) {
        return new Pow(sameDiff(), iX, false, i_y).outputVariable();
    }

    public SDVariable pow(SDVariable x, SDVariable y){
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.Pow(sameDiff(), x, y).outputVariable();
    }

    public SDVariable sqrt(SDVariable iX) {
        return new Sqrt(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable square(SDVariable iX) {
        return new Square(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable cube(SDVariable iX) {
        return new Cube(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable cubeDerivative(SDVariable iX) {
        return new CubeDerivative(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable floor(SDVariable iX) {
        return new Floor(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable floorDiv(SDVariable x, SDVariable y) {
        return new FloorDivOp(sameDiff(), x, y).outputVariable();
    }

    public List<SDVariable> floorDivBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new FloorDivBpOp(sameDiff(), x, y, grad).outputVariables());
    }

    public SDVariable floorMod(SDVariable x, SDVariable y) {
        return new FloorModOp(sameDiff(), x, y).outputVariable();
    }

    public List<SDVariable> floorModBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new FloorModBpOp(sameDiff(), x, y, grad).outputVariables());
    }

    public SDVariable ceil(SDVariable x) {
        return new Ceil(sameDiff(), x).outputVariable();
    }

    public SDVariable clipByValue(SDVariable x, double clipValueMin, double clipValueMax) {
        return new ClipByValue(sameDiff(), x, clipValueMin, clipValueMax).outputVariable();
    }

    public SDVariable clipByNorm(SDVariable x, double clipValue) {
        return new ClipByNorm(sameDiff(), x, clipValue).outputVariable();
    }

    public SDVariable clipByNorm(SDVariable x, double clipValue, int... dimensions) {
        return new ClipByNorm(sameDiff(), x, clipValue, dimensions).outputVariable();
    }

    public SDVariable relu(SDVariable iX, double cutoff) {
        return new RectifiedLinear(sameDiff(), iX, false, cutoff).outputVariable();
    }

    public SDVariable reluDerivative(SDVariable input, SDVariable grad){
        return new RectifiedLinearDerivative(sameDiff(), input, grad).outputVariable();
    }

    public SDVariable relu6(SDVariable iX, double cutoff) {
        return new Relu6(sameDiff(), iX, false, cutoff).outputVariable();
    }

    public SDVariable relu6Derivative(SDVariable iX, SDVariable wrt, double cutoff) {
        return new Relu6Derivative(sameDiff(), iX, wrt, cutoff).outputVariable();
    }

    public SDVariable softmax(SDVariable iX) {
        return new SoftMax(sameDiff(), new SDVariable[]{iX}).outputVariable();
    }

    public SDVariable softmax(SDVariable iX, int dimension) {
        return new SoftMax(sameDiff(), new SDVariable[]{iX}, dimension).outputVariable();
    }


    public SDVariable hardTanh(SDVariable iX) {
        return new HardTanh(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable hardTanhDerivative(SDVariable iX) {
        return new HardTanhDerivative(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable hardSigmoid(SDVariable in) {
        return new HardSigmoid(sameDiff(), in, false).outputVariable();
    }


    public SDVariable sigmoid(SDVariable iX) {
        return new Sigmoid(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable sigmoidDerivative(SDVariable iX, SDVariable wrt) {
        return new SigmoidDerivative(sameDiff(), iX, wrt).outputVariable();
    }


    public SDVariable logSigmoid(SDVariable iX) {
        return new LogSigmoid(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable powDerivative(SDVariable iX, double pow) {
        return new PowDerivative(sameDiff(), iX, false, pow).outputVariable();
    }


    public SDVariable swish(SDVariable iX) {
        return new Swish(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable swishDerivative(SDVariable iX) {
        return new SwishDerivative(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable gelu(SDVariable iX, boolean precise) {
        if (precise)
            return new PreciseGELU(sameDiff(), iX, false, precise).outputVariable();
        else
            return new GELU(sameDiff(), iX, false, precise).outputVariable();
    }

    public SDVariable geluDerivative(SDVariable iX, boolean precise) {
        if (precise)
            return new PreciseGELUDerivative(sameDiff(), iX, false, precise).outputVariable();
        else
            return new GELUDerivative(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable sign(SDVariable iX) {
        return new Sign(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable expandDims(SDVariable iX, int axis) {
        return new ExpandDims(sameDiff(), new SDVariable[]{iX}, axis).outputVariable();
    }

    public SDVariable squeeze(SDVariable iX, int... axis) {
        return new Squeeze(sameDiff(), iX, axis).outputVariable();
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, DataType dataType) {
        return new ConfusionMatrix(sameDiff(), labels, pred, dataType).outputVariable();
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses) {
        return new ConfusionMatrix(sameDiff(), labels, pred, numClasses).outputVariable();
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, SDVariable weights) {
        return new ConfusionMatrix(sameDiff(), labels, pred, weights).outputVariable();
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights) {
        return new ConfusionMatrix(sameDiff(), labels, pred, numClasses, weights).outputVariable();
    }

    public SDVariable matrixDeterminant(SDVariable in){
        return new MatrixDeterminant(sameDiff(), in, false).outputVariable();
    }

    public SDVariable matrixInverse(SDVariable in){
        return new MatrixInverse(sameDiff(), in, false).outputVariable();
    }

    public SDVariable onehot(SDVariable indices, int depth, int axis, double on, double off, DataType dataType) {
        return new OneHot(sameDiff(), indices, depth, axis, on, off, dataType).outputVariable();
    }

    public SDVariable onehot(SDVariable indices, int depth) {
        return new OneHot(sameDiff(), indices, depth).outputVariable();
    }

    public SDVariable reciprocal(SDVariable a) {
        return new Reciprocal(sameDiff(), a, false).outputVariable();
    }


    public SDVariable repeat(SDVariable iX, int axis) {
        return new Repeat(sameDiff(), new SDVariable[]{iX}, axis).outputVariable();

    }

    public SDVariable stack(SDVariable[] values, int axis) {
        return new Stack(sameDiff(), values, axis).outputVariable();
    }

    public SDVariable parallel_stack(SDVariable[] values) {
        return new ParallelStack(sameDiff(), values).outputVariable();
    }

    public SDVariable[] unstack(SDVariable value, int axis) {
        return new Unstack(sameDiff(), value, axis).outputVariables();
    }

    public SDVariable[] unstack(SDVariable value, int axis, int num) {
        return new Unstack(sameDiff(), value, axis, num).outputVariables();
    }

    public SDVariable assign(SDVariable x, SDVariable y) {
        return new Assign(sameDiff(), x, y).outputVariable();
    }

    public SDVariable assign(SDVariable x, Number num) {
        return new ScalarSet(sameDiff(), x, num).outputVariable();
    }


    public SDVariable softsign(SDVariable iX) {
        return new SoftSign(sameDiff(), iX, false).outputVariable();

    }


    public SDVariable softsignDerivative(SDVariable iX) {
        return new SoftSignDerivative(sameDiff(), iX, false).outputVariable();

    }


    public SDVariable softplus(SDVariable iX) {
        return new SoftPlus(sameDiff(), iX, false).outputVariable();

    }


    public SDVariable elu(SDVariable iX) {
        return new ELU(sameDiff(), iX, false).outputVariable();

    }


    public SDVariable eluDerivative(SDVariable iX) {
        return new ELUDerivative(sameDiff(), iX, false).outputVariable();

    }


    public SDVariable leakyRelu(SDVariable iX, double alpha) {
        return new LeakyReLU(sameDiff(), iX, false, alpha).outputVariable();

    }

    public SDVariable leakyReluDerivative(SDVariable iX, double cutoff) {
        return new LeakyReLUDerivative(sameDiff(), iX, false, cutoff).outputVariable();
    }


    public SDVariable reshape(SDVariable iX, int[] shape) {
        return new Reshape(sameDiff(), iX, ArrayUtil.toLongArray(shape)).outputVariable();
    }

    public SDVariable reshape(SDVariable iX, long[] shape) {
        return new Reshape(sameDiff(), iX, shape).outputVariable();
    }

    public SDVariable reshape(SDVariable iX, SDVariable shape) {
        return new Reshape(sameDiff(), iX, shape).outputVariable();
    }

    public SDVariable reverse(SDVariable x, int... dimensions) {
        return new Reverse(sameDiff(), x, dimensions).outputVariable();
    }

    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths, int seq_dim, int batch_dim) {
        return new ReverseSequence(sameDiff(), x, seq_lengths, seq_dim, batch_dim).outputVariable();
    }

    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths) {
        return new ReverseSequence(sameDiff(), x, seq_lengths).outputVariable();
    }

    public SDVariable sequenceMask(SDVariable lengths, SDVariable maxLen, DataType dataType) {
        return new SequenceMask(sameDiff(), lengths, maxLen, dataType).outputVariable();
    }

    public SDVariable sequenceMask(SDVariable lengths, int maxLen, DataType dataType) {
        return new SequenceMask(sameDiff(), lengths, maxLen, dataType).outputVariable();
    }

    public SDVariable sequenceMask(SDVariable lengths, DataType dataType) {
        return new SequenceMask(sameDiff(), lengths, dataType).outputVariable();
    }

    public SDVariable concat(int dimension, SDVariable... inputs) {
        return new Concat(sameDiff(), dimension, inputs).outputVariable();
    }

    public SDVariable fill(SDVariable shape, DataType dataType, double value) {
        return new Fill(sameDiff(), shape, dataType, value).outputVariable();
    }

    public SDVariable dot(SDVariable x, SDVariable y, int... dimensions) {
        return new Dot(sameDiff(), x, y, dimensions).outputVariable();
    }

    public SDVariable[] dotBp(SDVariable in1, SDVariable in2, SDVariable grad, boolean keepDims, int... dimensions) {
        return new DotBp(sameDiff(), in1, in2, grad, keepDims, dimensions).outputVariables();
    }

    public SDVariable cosineSimilarity(SDVariable iX, SDVariable i_y, int... dimensions) {
        return new CosineSimilarity(sameDiff(), iX, i_y, dimensions).outputVariable();
    }

    public SDVariable cosineDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return new CosineDistance(sameDiff(), ix, iy, dimensions).outputVariable();
    }


    public SDVariable euclideanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return new EuclideanDistance(sameDiff(), iX, i_y, dimensions).outputVariable();
    }


    public SDVariable manhattanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return new ManhattanDistance(sameDiff(), iX, i_y, dimensions).outputVariable();
    }

    public SDVariable hammingDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return new HammingDistance(sameDiff(), ix, iy, dimensions).outputVariable();
    }

    public SDVariable jaccardDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return new JaccardDistance(sameDiff(), ix, iy, dimensions).outputVariable();
    }

    public SDVariable weightedCrossEntropyWithLogits(SDVariable targets, SDVariable inputs, SDVariable weights) {
        return new WeightedCrossEntropyLoss(sameDiff(), targets, inputs, weights).outputVariable();
    }

    public SDVariable lossL2(SDVariable var){
        return new L2Loss(sameDiff(), var).outputVariable();
    }

    public SDVariable lossAbsoluteDifference(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new AbsoluteDifferenceLoss(sameDiff(), lossReduce, predictions, weights, label).outputVariable();
    }

    public SDVariable[] lossAbsoluteDifferenceBP(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new AbsoluteDifferenceLossBp(sameDiff(), lossReduce, predictions, weights, label).outputVariables();
    }

    public SDVariable lossCosineDistance(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce, int dimension){
        return new CosineDistanceLoss(sameDiff(), lossReduce, predictions, weights, label, dimension).outputVariable();
    }

    public SDVariable[] lossCosineDistanceBp(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce, int dimension){
        return new CosineDistanceLossBp(sameDiff(), lossReduce, predictions, weights, label, dimension).outputVariables();
    }

    public SDVariable lossHinge(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new HingeLoss(sameDiff(), lossReduce, predictions, weights, label).outputVariable();
    }

    public SDVariable[] lossHingeBp(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new HingeLossBp(sameDiff(), lossReduce, predictions, weights, label).outputVariables();
    }

    public SDVariable lossHuber(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce, double delta){
        return new HuberLoss(sameDiff(), lossReduce, predictions, weights, label, delta).outputVariable();
    }

    public SDVariable[] lossHuberBp(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce, double delta){
        return new HuberLossBp(sameDiff(), lossReduce, predictions, weights, label, delta).outputVariables();
    }

    public SDVariable lossLog(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce, double epsilon){
        return new LogLoss(sameDiff(), lossReduce, predictions, weights, label, epsilon).outputVariable();
    }

    public SDVariable[] lossLogBp(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce, double epsilon){
        return new LogLossBp(sameDiff(), lossReduce, predictions, weights, label, epsilon).outputVariables();
    }

    public SDVariable lossLogPoisson(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new LogPoissonLoss(sameDiff(), lossReduce, predictions, weights, label).outputVariable();
    }

    public SDVariable[] lossLogPoissonBp(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new LogPoissonLossBp(sameDiff(), lossReduce, predictions, weights, label).outputVariables();
    }

    public SDVariable lossLogPoissonFull(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new LogPoissonLoss(sameDiff(), lossReduce, predictions, weights, label, true).outputVariable();
    }

    public SDVariable[] lossLogPoissonFullBp(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new LogPoissonLossBp(sameDiff(), lossReduce, predictions, weights, label, true).outputVariables();
    }

    public SDVariable lossMeanPairwiseSquaredError(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new MeanPairwiseSquaredErrorLoss(sameDiff(), lossReduce, predictions, weights, label).outputVariable();
    }

    public SDVariable[] lossMeanPairwiseSquaredErrorBp(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new MeanPairwiseSquaredErrorLossBp(sameDiff(), lossReduce, predictions, weights, label).outputVariables();
    }

    public SDVariable lossMeanSquaredError(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new MeanSquaredErrorLoss(sameDiff(), lossReduce, predictions, weights, label).outputVariable();
    }

    public SDVariable[] lossMeanSquaredErrorBp(SDVariable label, SDVariable predictions, SDVariable weights, LossReduce lossReduce){
        return new MeanSquaredErrorLossBp(sameDiff(), lossReduce, predictions, weights, label).outputVariables();
    }

    public SDVariable lossSigmoidCrossEntropy(SDVariable labels, SDVariable logits, SDVariable weights, LossReduce lossReduce, double labelSmoothing) {
        return new SigmoidCrossEntropyLoss(sameDiff(), lossReduce, logits, weights, labels, labelSmoothing).outputVariable();
    }

    public SDVariable[] lossSigmoidCrossEntropyBp(SDVariable labels, SDVariable logits, SDVariable weights, LossReduce lossReduce, double labelSmoothing) {
        return new SigmoidCrossEntropyLossBp(sameDiff(), lossReduce, logits, weights, labels, labelSmoothing).outputVariables();
    }

    public SDVariable lossSoftmaxCrossEntropy(SDVariable labels, SDVariable logits, SDVariable weights, LossReduce lossReduce, double labelSmoothing) {
        return new SoftmaxCrossEntropyLoss(sameDiff(), lossReduce, logits, weights, labels, labelSmoothing).outputVariable();
    }

    public SDVariable[] lossSoftmaxCrossEntropyBp(SDVariable labels, SDVariable logits, SDVariable weights, LossReduce lossReduce, double labelSmoothing) {
        return new SoftmaxCrossEntropyLossBp(sameDiff(), lossReduce, logits, weights, labels, labelSmoothing).outputVariables();
    }

    public SDVariable lossSoftmaxCrossEntropyWithLogits(SDVariable labels, SDVariable logits, SDVariable weights, int classDim) {
        return new SoftmaxCrossEntropyWithLogitsLoss(sameDiff(), logits, weights, labels, classDim).outputVariable();
    }

    public SDVariable[] lossSoftmaxCrossEntropyWithLogitsBp(SDVariable labels, SDVariable logits, SDVariable weights, int classDim) {
        return new SoftmaxCrossEntropyWithLogitsLossBp(sameDiff(), logits, weights, labels, classDim).outputVariables();
    }

    public SDVariable lossSparseSoftmaxCrossEntropy(SDVariable logits, SDVariable labels){
        return new SparseSoftmaxCrossEntropyLossWithLogits(sameDiff(), logits, labels).outputVariable();
    }

    public SDVariable[] lossSparseSoftmaxCrossEntropyBp(SDVariable logits, SDVariable labels){
        return new SparseSoftmaxCrossEntropyLossWithLogitsBp(sameDiff(), logits, labels).outputVariables();
    }


    public SDVariable xwPlusB(SDVariable input, SDVariable weights, SDVariable bias) {
        return new XwPlusB(sameDiff(), input, weights, bias).outputVariable();
    }

    public SDVariable reluLayer(SDVariable input, SDVariable weights, SDVariable bias) {
        return new ReluLayer(sameDiff(), input, weights, bias).outputVariable();
    }

    public SDVariable mmul(SDVariable x,
                           SDVariable y,
                           MMulTranspose mMulTranspose) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return new Mmul(sameDiff(), x, y, mMulTranspose).outputVariable();
    }


    public SDVariable mmul(SDVariable x,
                           SDVariable y) {
        return mmul(x, y, MMulTranspose.allFalse());
    }

    public List<SDVariable> mmulBp(SDVariable x, SDVariable y, SDVariable eps, MMulTranspose mt) {
        return Arrays.asList(new MmulBp(sameDiff(), x, y, eps, mt).outputVariables());
    }

    public SDVariable[] batchMmul(SDVariable[] matricesA,
                                SDVariable[] matricesB) {
        return batchMmul(matricesA, matricesB, false, false);
    }


    public SDVariable[] batchMmul(SDVariable[] matricesA,
                                SDVariable[] matricesB,
                                boolean transposeA,
                                boolean transposeB) {
        return batchMmul(ArrayUtils.addAll(matricesA, matricesB), transposeA, transposeB);
    }


    public SDVariable[] batchMmul(SDVariable[] matrices,
                                boolean transposeA,
                                boolean transposeB) {
        return new BatchMmul(sameDiff(), matrices, transposeA, transposeB).outputVariables();
    }


    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return new TensorMmul(sameDiff(), x, y, dimensions).outputVariable();
    }

    public SDVariable dotProductAttention(SDVariable queries, SDVariable keys, SDVariable values, SDVariable mask, boolean scaled) {
        return new DotProductAttention(sameDiff(), queries, keys, values, mask, scaled, false).outputVariable();
    }

    public List<SDVariable> dotProductAttention(SDVariable queries, SDVariable keys, SDVariable values, SDVariable mask, boolean scaled, boolean withWeights) {
        return Arrays.asList(new DotProductAttention(sameDiff(), queries, keys, values, mask, scaled, withWeights).outputVariables());
    }

    public List<SDVariable> dotProductAttentionBp(SDVariable queries, SDVariable keys, SDVariable values, SDVariable gradient, SDVariable mask,  boolean scaled) {
        return Arrays.asList(new DotProductAttentionBp(sameDiff(), queries, keys, values, gradient, mask, scaled).outputVariables());
    }

    public SDVariable multiHeadDotProductAttention(SDVariable queries, SDVariable keys, SDVariable values, SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo, SDVariable mask, boolean scaled) {
        return new MultiHeadDotProductAttention(sameDiff(), queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled, false).outputVariable();
    }

    public List<SDVariable> multiHeadDotProductAttention(SDVariable queries, SDVariable keys, SDVariable values,SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo, SDVariable mask, boolean scaled, boolean withWeights) {
        return Arrays.asList(new MultiHeadDotProductAttention(sameDiff(), queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled, withWeights).outputVariables());
    }

    public List<SDVariable> multiHeadDotProductAttentionBp(SDVariable queries, SDVariable keys, SDVariable values,SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo, SDVariable gradient, SDVariable mask,  boolean scaled) {
        return Arrays.asList(new MultiHeadDotProductAttentionBp(sameDiff(), queries, keys, values, Wq, Wk, Wv, Wo, gradient, mask, scaled).outputVariables());
    }

    public SDVariable softmaxDerivative(SDVariable functionInput, SDVariable wrt, Integer dimension) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new SoftmaxBp(sameDiff(), functionInput, wrt, dimension).outputVariable();
    }


    public SDVariable logSoftmax(SDVariable i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return new LogSoftMax(sameDiff(), i_v).outputVariable();

    }


    public SDVariable logSoftmax(SDVariable i_v, int dimension) {
        validateDifferentialFunctionsameDiff(i_v);
        return new LogSoftMax(sameDiff(), i_v, dimension).outputVariable();

    }


    public SDVariable logSoftmaxDerivative(SDVariable arg, SDVariable wrt) {
        validateDifferentialFunctionsameDiff(arg);
        return new LogSoftMaxDerivative(sameDiff(), arg, wrt).outputVariable();
    }


    public SDVariable logSoftmaxDerivative(SDVariable arg, SDVariable wrt, int dimension) {
        validateDifferentialFunctionsameDiff(arg);
        return new LogSoftMaxDerivative(sameDiff(), arg, wrt, dimension).outputVariable();
    }

    public SDVariable logSumExp(SDVariable arg, boolean keepDims, int... dimension) {
        return new LogSumExp(sameDiff(), arg, keepDims, dimension).outputVariable();
    }


    public SDVariable selu(SDVariable arg) {
        validateDifferentialFunctionsameDiff(arg);
        return new SELU(sameDiff(), arg, false).outputVariable();
    }


    public SDVariable seluDerivative(SDVariable arg) {
        validateDifferentialFunctionsameDiff(arg);
        return new SELUDerivative(sameDiff(), arg, false).outputVariable();
    }


    public SDVariable rsub(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RSubOp(sameDiff(), differentialFunction, i_v).outputVariable();
    }

    public List<SDVariable> rsubBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new RSubBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable rdiv(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RDivOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariable();
    }

    public List<SDVariable> rdivBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new RDivBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable rdivi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RDivOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariable();
    }


    public SDVariable rsubi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RSubOp(sameDiff(), differentialFunction, i_v).outputVariable();
    }

    public SDVariable add(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new AddOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariable();

    }

    public SDVariable mergeAdd(SDVariable... differentialFunctions) {
        for (SDVariable df : differentialFunctions)
            validateDifferentialFunctionsameDiff(df);
        return new MergeAddOp(sameDiff(), differentialFunctions, false).outputVariable();
    }

    public SDVariable mergeMax(SDVariable... differentialFunctions) {
        for (SDVariable df : differentialFunctions)
            validateDifferentialFunctionsameDiff(df);
        return new MergeMax(sameDiff(), differentialFunctions).outputVariable();
    }

    public SDVariable mergeAvg(SDVariable... differentialFunctions) {
        for (SDVariable df : differentialFunctions)
            validateDifferentialFunctionsameDiff(df);
        return new MergeAvg(sameDiff(), differentialFunctions).outputVariable();
    }

    public SDVariable diag(SDVariable sdVariable) {
        validateDifferentialFunctionsameDiff(sdVariable);
        return new Diag(sameDiff(), new SDVariable[]{sdVariable}, false).outputVariable();
    }

    public SDVariable diagPart(SDVariable sdVariable) {
        validateDifferentialFunctionsameDiff(sdVariable);
        return new DiagPart(sameDiff(), new SDVariable[]{sdVariable}, false).outputVariable();
    }

    public SDVariable setDiag(SDVariable in, SDVariable diag) {
        return new MatrixSetDiag(sameDiff(), in, diag, false).outputVariable();
    }


    public SDVariable batchToSpace(SDVariable differentialFunction, int[] blocks, int[][] crops) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new BatchToSpace(sameDiff(), new SDVariable[]{differentialFunction}, blocks, crops, false)
                .outputVariable();
    }

    public SDVariable spaceToBatch(SDVariable differentialFunction, int[] blocks, int[][] padding) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SpaceToBatch(sameDiff(), new SDVariable[]{differentialFunction}, blocks, padding, false)
                .outputVariable();
    }

    public SDVariable depthToSpace(SDVariable differentialFunction, int blocksSize, String dataFormat) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DepthToSpace(sameDiff(), new SDVariable[]{differentialFunction}, blocksSize, dataFormat)
                .outputVariable();
    }

    public SDVariable spaceToDepth(SDVariable differentialFunction, int blocksSize, String dataFormat) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SpaceToDepth(sameDiff(), new SDVariable[]{differentialFunction}, blocksSize, dataFormat)
                .outputVariable();
    }

    public SDVariable[] dynamicPartition(SDVariable differentialFunction, SDVariable partitions, int numPartitions) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DynamicPartition(sameDiff(), differentialFunction, partitions, numPartitions)
                .outputVariables();
    }

    public SDVariable[] dynamicPartitionBp(SDVariable input, SDVariable partitions, SDVariable[] grads, int numPartitions){
        return new DynamicPartitionBp(sameDiff(), input, partitions, grads, numPartitions).outputVariables();
    }

    public SDVariable dynamicStitch(SDVariable[] indices, SDVariable[] differentialFunctions) {
        for (SDVariable df : differentialFunctions)
            validateDifferentialFunctionsameDiff(df);

        return new DynamicStitch(sameDiff(), indices, differentialFunctions).outputVariable();
    }

    public SDVariable segmentMax(SDVariable data, SDVariable segmentIds){
        return new SegmentMax(sameDiff(), data, segmentIds).outputVariable();
    }

    public SDVariable[] segmentMaxBp(SDVariable data, SDVariable segmentIds, SDVariable gradient){
        return new SegmentMaxBp(sameDiff(), data, segmentIds, gradient).outputVariables();
    }

    public SDVariable segmentMin(SDVariable data, SDVariable segmentIds){
        return new SegmentMin(sameDiff(), data, segmentIds).outputVariable();
    }

    public SDVariable[] segmentMinBp(SDVariable data, SDVariable segmentIds, SDVariable gradient){
        return new SegmentMinBp(sameDiff(), data, segmentIds, gradient).outputVariables();
    }

    public SDVariable segmentMean(SDVariable data, SDVariable segmentIds){
        return new SegmentMean(sameDiff(), data, segmentIds).outputVariable();
    }

    public SDVariable[] segmentMeanBp(SDVariable data, SDVariable segmentIds, SDVariable gradient){
        return new SegmentMeanBp(sameDiff(), data, segmentIds, gradient).outputVariables();
    }

    public SDVariable segmentProd(SDVariable data, SDVariable segmentIds){
        return new SegmentProd(sameDiff(), data, segmentIds).outputVariable();
    }

    public SDVariable[] segmentProdBp(SDVariable data, SDVariable segmentIds, SDVariable gradient){
        return new SegmentProdBp(sameDiff(), data, segmentIds, gradient).outputVariables();
    }

    public SDVariable segmentSum(SDVariable data, SDVariable segmentIds){
        return new SegmentSum(sameDiff(), data, segmentIds).outputVariable();
    }

    public SDVariable[] segmentSumBp(SDVariable data, SDVariable segmentIds, SDVariable gradient){
        return new SegmentSumBp(sameDiff(), data, segmentIds, gradient).outputVariables();
    }


    public SDVariable unsortedSegmentMax(SDVariable data, SDVariable segmentIds, int numSegments){
        return new UnsortedSegmentMax(sameDiff(), data, segmentIds, numSegments).outputVariable();
    }

    public SDVariable[] unsortedSegmentMaxBp(SDVariable data, SDVariable segmentIds, SDVariable gradient, int numSegments){
        return new UnsortedSegmentMaxBp(sameDiff(), data, segmentIds, gradient, numSegments).outputVariables();
    }

    public SDVariable unsortedSegmentMin(SDVariable data, SDVariable segmentIds, int numSegments){
        return new UnsortedSegmentMin(sameDiff(), data, segmentIds, numSegments).outputVariable();
    }

    public SDVariable[] unsortedSegmentMinBp(SDVariable data, SDVariable segmentIds, SDVariable gradient, int numSegments){
        return new UnsortedSegmentMinBp(sameDiff(), data, segmentIds, gradient, numSegments).outputVariables();
    }

    public SDVariable unsortedSegmentMean(SDVariable data, SDVariable segmentIds, int numSegments){
        return new UnsortedSegmentMean(sameDiff(), data, segmentIds, numSegments).outputVariable();
    }

    public SDVariable[] unsortedSegmentMeanBp(SDVariable data, SDVariable segmentIds, SDVariable gradient, int numSegments){
        return new UnsortedSegmentMeanBp(sameDiff(), data, segmentIds, gradient, numSegments).outputVariables();
    }

    public SDVariable unsortedSegmentProd(SDVariable data, SDVariable segmentIds, int numSegments){
        return new UnsortedSegmentProd(sameDiff(), data, segmentIds, numSegments).outputVariable();
    }

    public SDVariable[] unsortedSegmentProdBp(SDVariable data, SDVariable segmentIds, SDVariable gradient, int numSegments){
        return new UnsortedSegmentProdBp(sameDiff(), data, segmentIds, gradient, numSegments).outputVariables();
    }

    public SDVariable unsortedSegmentSum(SDVariable data, SDVariable segmentIds, int numSegments){
        return new UnsortedSegmentSum(sameDiff(), data, segmentIds, numSegments).outputVariable();
    }

    public SDVariable[] unsortedSegmentSumBp(SDVariable data, SDVariable segmentIds, SDVariable gradient, int numSegments){
        return new UnsortedSegmentSumBp(sameDiff(), data, segmentIds, gradient, numSegments).outputVariables();
    }

    public SDVariable unsortedSegmentSqrtN(SDVariable data, SDVariable segmentIds, int numSegments){
        return new UnsortedSegmentSqrtN(sameDiff(), data, segmentIds, numSegments).outputVariable();
    }

    public SDVariable[] unsortedSegmentSqrtNBp(SDVariable data, SDVariable segmentIds, SDVariable gradient, int numSegments){
        return new UnsortedSegmentSqrtNBp(sameDiff(), data, segmentIds, gradient, numSegments).outputVariables();
    }




    public SDVariable dilation2D(SDVariable df, SDVariable weights, int[] strides,
                                 int[] rates, boolean isSameMode) {
        validateDifferentialFunctionsameDiff(df);
        return new Dilation2D(sameDiff(), new SDVariable[]{df, weights}, strides, rates, isSameMode, false)
                .outputVariable();
    }

    public SDVariable shape(SDVariable df) {
        validateDifferentialFunctionsameDiff(df);
        return new org.nd4j.linalg.api.ops.impl.shape.Shape(sameDiff(), df, false).outputVariable();
    }

    public SDVariable size(SDVariable in) {
        return new Size(sameDiff(), in).outputVariable();
    }

    public SDVariable sizeAt(SDVariable in, int dimension){
        return new SizeAt(sameDiff(), in, dimension).outputVariable();
    }

    public SDVariable rank(SDVariable df) {
        return new Rank(sameDiff(), df, false).outputVariable();
    }

    public SDVariable gather(SDVariable df, int[] indices, int axis) {
        validateDifferentialFunctionsameDiff(df);
        return new Gather(sameDiff(), df, indices, axis, false).outputVariable();
    }

    public SDVariable gather(SDVariable df, SDVariable indices, int axis) {
        validateDifferentialFunctionsameDiff(df);
        return new Gather(sameDiff(), df, indices, axis, false).outputVariable();
    }

    public SDVariable gatherNd(SDVariable df, SDVariable indices) {
        validateDifferentialFunctionsameDiff(df);
        return new GatherNd(sameDiff(), df, indices, false).outputVariable();
    }

    public SDVariable trace(SDVariable in){
        return new Trace(sameDiff(), in).outputVariable();
    }

    public SDVariable cross(SDVariable a, SDVariable b) {
        validateDifferentialFunctionsameDiff(a);
        return new Cross(sameDiff(), new SDVariable[]{a, b}).outputVariable();
    }

    public SDVariable erf(SDVariable differentialFunction) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new Erf(sameDiff(), differentialFunction, false).outputVariable();
    }

    public SDVariable erfc(SDVariable differentialFunction) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new Erfc(sameDiff(), differentialFunction, false).outputVariable();
    }

    public SDVariable addi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new AddOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariable();
    }

    public List<SDVariable> addBp(SDVariable x, SDVariable y, SDVariable grad) {
        SDVariable[] ret = new AddBpOp(sameDiff(), x, y, grad).outputVariables();
        return Arrays.asList(ret);
    }


    public SDVariable sub(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SubOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariable();
    }

    public SDVariable squaredDifference(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SquaredDifferenceOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false)
                .outputVariable();
    }


    public List<SDVariable> subBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new SubBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable subi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SubOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariable();

    }


    public SDVariable mul(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new MulOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariable();
    }

    public List<SDVariable> mulBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new MulBpOp(sameDiff(), x, y, grad).outputVariables());
    }

    public List<SDVariable> modBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new ModBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable muli(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new MulOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariable();
    }

    public SDVariable mod(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ModOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariable();
    }

    public SDVariable div(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DivOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariable();
    }

    public SDVariable truncatedDiv(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new TruncateDivOp(sameDiff(), differentialFunction, i_v, false).outputVariable();
    }

    public List<SDVariable> divBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new DivBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable divi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DivOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariable();
    }


    public SDVariable rsub(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseSubtraction(sameDiff(), differentialFunction, i_v).outputVariable();

    }


    public SDVariable rdiv(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseDivision(sameDiff(), differentialFunction, i_v).outputVariable();

    }


    public SDVariable rdivi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseDivision(sameDiff(), differentialFunction, i_v, true).outputVariable();
    }


    public SDVariable rsubi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseSubtraction(sameDiff(), differentialFunction, i_v, true).outputVariable();

    }


    public SDVariable add(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarAdd(sameDiff(), differentialFunction, i_v, false).outputVariable();
    }


    public SDVariable addi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarAdd(sameDiff(), differentialFunction, i_v, true).outputVariable();
    }


    public SDVariable sub(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarSubtraction(sameDiff(), differentialFunction, i_v).outputVariable();
    }


    public SDVariable subi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarSubtraction(sameDiff(), differentialFunction, i_v, true).outputVariable();

    }


    public SDVariable mul(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarMultiplication(sameDiff(), differentialFunction, i_v).outputVariable();

    }


    public SDVariable muli(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarMultiplication(sameDiff(), differentialFunction, i_v, true).outputVariable();

    }


    public SDVariable div(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarDivision(sameDiff(), differentialFunction, i_v).outputVariable();
    }


    public SDVariable divi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarDivision(sameDiff(), differentialFunction, i_v, true).outputVariable();
    }


    public SDVariable gt(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThan(sameDiff(), new SDVariable[]{functionInput, functionInput1}, false).outputVariable();
    }


    public SDVariable lt(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThan(sameDiff(), new SDVariable[]{functionInput, functionInput1}, false).outputVariable();
    }


    public SDVariable gti(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThan(sameDiff(), new SDVariable[]{functionInput, functionInput1}, true).outputVariable();
    }


    public SDVariable lti(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThan(sameDiff(), new SDVariable[]{functionInput, functionInput1}, true).outputVariable();
    }


    public SDVariable gte(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThanOrEqual(sameDiff(), new SDVariable[]{functionInput, functionInput1}, false).outputVariable();
    }


    public SDVariable lte(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThanOrEqual(sameDiff(), new SDVariable[]{functionInput, functionInput1}, false).outputVariable();
    }


    public SDVariable gtei(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThanOrEqual(sameDiff(), new SDVariable[]{functionInput, functionInput1}, true).outputVariable();
    }


    public SDVariable ltOrEqi(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThanOrEqual(sameDiff(), new SDVariable[]{functionInput, functionInput1}, true).outputVariable();
    }


    public SDVariable gt(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThan(sameDiff(), functionInput, functionInput1, false).outputVariable();
    }


    public SDVariable lt(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThan(sameDiff(), functionInput, functionInput1, false).outputVariable();
    }


    public SDVariable gti(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThan(sameDiff(), functionInput, functionInput1, true).outputVariable();
    }


    public SDVariable lti(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThan(sameDiff(), functionInput, functionInput1, true).outputVariable();
    }


    public SDVariable gte(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThanOrEqual(sameDiff(), functionInput, functionInput1, false).outputVariable();
    }


    public SDVariable lte(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThanOrEqual(sameDiff(), functionInput, functionInput1, false).outputVariable();
    }


    public SDVariable gtei(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThanOrEqual(sameDiff(), functionInput, functionInput1, true).outputVariable();
    }


    public SDVariable ltei(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThanOrEqual(sameDiff(), functionInput, functionInput1, true).outputVariable();
    }


    public SDVariable eq(SDVariable iX, double i_y) {
        return new ScalarEquals(sameDiff(), iX, i_y).outputVariable();
    }

    public SDVariable eqi(SDVariable iX, double i_y) {
        return new ScalarEquals(sameDiff(), iX, i_y, true).outputVariable();
    }

    public SDVariable isNonDecreasing(SDVariable iX) {
        validateDifferentialFunctionsameDiff(iX);
        return new IsNonDecreasing(sameDiff(), new SDVariable[]{iX}, false).outputVariable();
    }

    public SDVariable isStrictlyIncreasing(SDVariable iX) {
        validateDifferentialFunctionsameDiff(iX);
        return new IsStrictlyIncreasing(sameDiff(), new SDVariable[]{iX}, false).outputVariable();
    }

    public SDVariable isNumericTensor(SDVariable iX) {
        validateDifferentialFunctionsameDiff(iX);
        return new IsNumericTensor(sameDiff(), new SDVariable[]{iX}, false).outputVariable();
    }

    public SDVariable slice(SDVariable input, int[] begin, int[] size) {
        return new Slice(sameDiff(), input, begin, size).outputVariable();
    }

    public SDVariable slice(SDVariable input, SDVariable begin, SDVariable size) {
        return new Slice(sameDiff(), input, begin, size).outputVariable();
    }

    public SDVariable sliceBp(SDVariable input, SDVariable gradient, int[] begin, int[] size) {
        return new SliceBp(sameDiff(), input, gradient, begin, size).outputVariable();
    }

    public SDVariable sliceBp(SDVariable input, SDVariable gradient, SDVariable begin, SDVariable size) {
        return new SliceBp(sameDiff(), input, gradient, begin, size).outputVariable();
    }


    public SDVariable stridedSlice(SDVariable input, int[] begin, int[] end, int[] strides) {
        return new StridedSlice(sameDiff(), input, begin, end, strides).outputVariable();
    }

    public SDVariable stridedSlice(SDVariable input, long[] begin, long[] end, long[] strides) {
        return new StridedSlice(sameDiff(), input, begin, end, strides).outputVariable();
    }


    public SDVariable stridedSlice(SDVariable in, int[] begin, int[] end, int[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return new StridedSlice(sameDiff(), in, begin, end, strides, beginMask, endMask, ellipsisMask,
                newAxisMask, shrinkAxisMask).outputVariable();
    }

    public SDVariable stridedSlice(SDVariable in, long[] begin, long[] end, long[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return new StridedSlice(sameDiff(), in, begin, end, strides, beginMask, endMask, ellipsisMask,
                newAxisMask, shrinkAxisMask).outputVariable();
    }

    public SDVariable stridedSliceBp(SDVariable in, SDVariable grad, long[] begin, long[] end, long[] strides, int beginMask,
                                     int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return new StridedSliceBp(sameDiff(), in, grad, begin, end, strides, beginMask, endMask, ellipsisMask,
                newAxisMask, shrinkAxisMask).outputVariable();
    }

    public SDVariable stridedSliceBp(SDVariable in, SDVariable grad, SDVariable begin, SDVariable end, SDVariable strides, int beginMask,
                                     int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return new StridedSliceBp(sameDiff(), in, grad, begin, end, strides, beginMask, endMask, ellipsisMask,
                newAxisMask, shrinkAxisMask).outputVariable();
    }

    public SDVariable scatterAdd(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterAdd(sameDiff(), ref, indices, updates).outputVariable();
    }

    public SDVariable scatterSub(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterSub(sameDiff(), ref, indices, updates).outputVariable();
    }

    public SDVariable scatterMul(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterMul(sameDiff(), ref, indices, updates).outputVariable();
    }

    public SDVariable scatterDiv(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterDiv(sameDiff(), ref, indices, updates).outputVariable();
    }

    public SDVariable scatterMax(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterMax(sameDiff(), ref, indices, updates).outputVariable();
    }

    public SDVariable scatterMin(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterMin(sameDiff(), ref, indices, updates).outputVariable();
    }

    public SDVariable scatterUpdate(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterUpdate(sameDiff(), ref, indices, updates).outputVariable();
    }


    public SDVariable merge(SDVariable... inputs){
        return new Merge(sameDiff(), inputs).outputVariable();
    }

    public SDVariable[] switchOp(SDVariable input, SDVariable predicate){
        return new Switch(sameDiff(), input, predicate).outputVariables();
    }


    public void validateDifferentialFunctionsameDiff(
            SDVariable function) {

        Preconditions.checkState(function != null, "Passed in function was null.");
        Preconditions.checkState(function.getSameDiff() == sameDiff);

        Preconditions.checkState(function.getSameDiff() == this.getSameDiff(),
                "Function applications must be contained " +
                        "in same sameDiff. The left %s must match this function %s", function, this);
        Preconditions.checkState(sameDiff == this.getSameDiff(), "Function applications must be " +
                "contained in same sameDiff. The left %s must match this function ", function, this);
    }


    public void validateDifferentialFunctionGraph(SDVariable function) {
        Preconditions.checkState(function.getSameDiff() == this.getSameDiff(),
                "Function applications must be contained in same graph. The left %s must match this function %s",
                function, this);

    }


    /**
     * @param func
     * @param input
     * @return
     */
    public SDVariable doRepeat(SDVariable func,
                               SDVariable input) {
        validateDifferentialFunctionsameDiff(func);
        validateDifferentialFunctionsameDiff(input);

        // FIXME: int cast!
        return tile(func, ArrayUtil.toInts(input.getShape()));
    }

    public SDVariable enter(SDVariable x, String frameName){
        return new Enter(sameDiff, frameName, x).outputVariable();
    }

    public SDVariable enter(SDVariable x, String frameName, boolean isConstant){
        return new Enter(sameDiff, frameName, x, isConstant).outputVariable();
    }

    public SDVariable exit(SDVariable x){
        return new Exit(sameDiff, x).outputVariable();
    }

    public SDVariable nextIteration(SDVariable x){
        return new NextIteration(sameDiff, x).outputVariable();
    }


    public String toString() {
        return "DifferentialFunctionFactory{methodNames=" + methodNames + "}";
    }
}
