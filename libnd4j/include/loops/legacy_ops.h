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

//
// @author raver119@gmail.com
//

#ifndef PROJECT_LEGACY_OPS_H
#define PROJECT_LEGACY_OPS_H

#define AGGREGATE_OPS \
        (0, aggregateOps::HierarchicSoftmax) ,\
        (1, aggregateOps::Dot) ,\
        (2, aggregateOps::Axpy) ,\
        (3, aggregateOps::SkipGram) ,\
        (4, aggregateOps::CBOW) ,\
        (5, aggregateOps::GEMM)

#define BROADCAST_BOOL_OPS \
        (0, EqualTo),\
        (1, GreaterThan),\
        (2, LessThan),\
        (3, Epsilon),\
        (4, GreaterThanOrEqual),\
        (5, LessThanOrEqual),\
        (6, NotEqualTo),\
        (7, And),\
        (8, Or),\
        (9, Xor) ,\
        (10, Not)


#define BROADCAST_OPS \
       (0, Add), \
       (1, Subtract), \
       (2, Multiply), \
       (3, Divide), \
       (4, ReverseDivide), \
       (5, ReverseSubtract), \
       (6, Copy) ,\
       (13, MinPairwise) ,\
       (14, MaxPairwise) ,\
       (15, AMinPairwise) ,\
       (16, AMaxPairwise) ,\
       (17, SquaredSubtract),\
       (18, FloorMod),\
       (19, FloorDiv),\
       (20, ReverseMod),\
       (21, SafeDivide),\
       (22, Mod) ,\
       (23, TruncateDiv), \
       (26, Atan2) ,\
       (27, LogicalOr) ,\
       (28, LogicalXor) ,\
       (29, LogicalNot) ,\
       (30, LogicalAnd)

// these ops return same data type as input
#define TRANSFORM_SAME_OPS \
        (0,Abs), \
        (1,Sign), \
        (2,Ones), \
        (3, Neg), \
        (4, Round), \
        (5,TimesOneMinus), \
        (6,Cube), \
        (7,OneMinus), \
        (8,Col2Im), \
        (9,Im2col),\
        (11,Reciprocal), \
        (12,Square), \
        (15,Identity), \
        (17,Ceiling), \
        (18,Floor), \
        (19,ClipByValue) ,\
        (20,Reverse)


// these ops return bool
#define TRANSFORM_BOOL_OPS \
        (0,IsMax), \
        (1,IsInf), \
        (2,IsNan), \
        (3,IsFinite), \
        (4,IsInfOrNan), \
        (5,MatchConditionBool), \
        (7,IsPositive)

#define TRANSFORM_STRICT_OPS \
        (0,SoftMax), \
        (1,SoftMaxDerivative), \
        (2,LogSoftMax) ,\
        (3,ELUDerivative), \
        (4,TanhDerivative), \
        (5,HardTanhDerivative), \
        (6,SigmoidDerivative), \
        (7,SoftSignDerivative), \
        (8,TanDerivative) ,\
        (9,SELUDerivative) ,\
        (10,HardSigmoidDerivative) ,\
        (11,RationalTanhDerivative) ,\
        (12,RectifiedTanhDerivative) ,\
        (13,SwishDerivative) ,\
        (14,ACoshDerivative) ,\
        (15,ASinhDerivative) ,\
        (16,SinhDerivative), \
        (17,LogSigmoidDerivative) ,\
        (18,SpecialDerivative), \
        (19,Stabilize), \
        (20,StabilizeFP16) ,\
        (21,CubeDerivative)

// these ops return one of FLOAT data types
#define TRANSFORM_FLOAT_OPS \
        (0,Cosine), \
        (1,Exp), \
        (2,Log), \
        (3,SetRange), \
        (4,Sigmoid), \
        (5,Sin), \
        (6,SoftPlus), \
        (7,Sqrt), \
        (8,Tanh), \
        (9,ACos), \
        (10,ASin), \
        (11,ATan), \
        (12,HardTanh), \
        (13,SoftSign), \
        (14,ELU), \
        (15,Histogram), \
        (16,HardSigmoid), \
        (17,RationalTanh) ,\
        (18,RectifiedTanh) ,\
        (19,Sinh) ,\
        (20,Cosh) ,\
        (21,Tan) ,\
        (22,SELU) ,\
        (23,Pooling2D) ,\
        (24,Swish) ,\
        (25,RSqrt), \
        (26,Log1p), \
        (27,Erf), \
        (28,ACosh), \
        (29,ASinh), \
        (30,Rint), \
        (31,LogSigmoid), \
        (32,Erfc) ,\
        (33,Expm1), \
        (34,ATanh)




#define SUMMARY_STATS_OPS \
        (0, SummaryStatsVariance), \
        (1, SummaryStatsStandardDeviation)


#define SCALAR_BOOL_OPS \
        (0, EqualTo),\
        (1, GreaterThan),\
        (2, LessThan),\
        (3, Epsilon),\
        (4, GreaterThanOrEqual),\
        (5, LessThanOrEqual),\
        (6, NotEqualTo),\
        (7, And),\
        (8, Or),\
        (9, Xor) ,\
        (10, Not)

#define SCALAR_OPS \
        (0, Add),\
        (1, Subtract),\
        (2, Multiply),\
        (3, Divide),\
        (4, ReverseDivide),\
        (5, ReverseSubtract),\
        (6, MaxPairwise),\
        (13, MinPairwise),\
        (14, Copy),\
        (15, Mod),\
        (16, ReverseMod),\
        (17, Remainder),\
        (18, FMod) ,\
        (19, TruncateDiv) ,\
        (20, FloorDiv) ,\
        (21, FloorMod), \
        (22, SquaredSubtract),\
        (23, SafeDivide), \
        (24, AMaxPairwise), \
        (25, AMinPairwise), \
        (26, Atan2) ,\
        (27, LogicalOr) ,\
        (28, LogicalXor) ,\
        (29, LogicalNot) ,\
        (30, LogicalAnd) ,\
        (31, Pow) ,\
        (32, PowDerivative) ,\
        (33, CompareAndSet) ,\
        (34, SXELogitsSmoother), \
        (35, LeakyRELU), \
        (36, LeakyRELUDerivative), \
        (37, ReplaceNans) ,\
        (38, LogX) ,\
        (39, RELU), \
        (40, RELU6), \
        (41, Step), \
        (42, LstmClip)






#define REDUCE3_OPS \
        (0, ManhattanDistance), \
        (1, EuclideanDistance), \
        (2, CosineSimilarity), \
        (3, Dot), \
        (4, EqualsWithEps) ,\
        (5, CosineDistance) ,\
        (6, JaccardDistance) ,\
        (7, SimpleHammingDistance)

#define REDUCE_LONG_OPS \
        (0, CountNonZero), \
        (1, CountZero), \
        (2, MatchCondition)

#define REDUCE_BOOL_OPS \
        (0, Any) ,\
        (1, All), \
        (2, IsFinite), \
        (3, IsInfOrNan), \
        (4, IsNan), \
        (5, IsInf), \
        (6, IsPositive)

#define REDUCE_SAME_OPS \
        (0, Sum), \
        (1, Max), \
        (2, Min), \
        (3, Prod), \
        (4, ASum), \
        (5, AMax) ,\
        (6, AMin)



#define REDUCE_FLOAT_OPS \
        (0, Mean), \
        (1, AMean) ,\
        (2, Norm1), \
        (3, Norm2), \
        (4, NormMax), \
        (5, NormFrobenius), \
        (6, NormP), \
        (7, SquaredNorm) ,\
        (8, Entropy) ,\
        (9, LogEntropy) ,\
        (10, ShannonEntropy) ,\
        (11, LogSumExp)




#define RANDOM_OPS \
        (0, UniformDistribution) ,\
        (1, DropOut) ,\
        (2, DropOutInverted) ,\
        (3, ProbablisticMerge) ,\
        (4, Linspace) ,\
        (5, Choice) ,\
        (6, GaussianDistribution) ,\
        (7, BernoulliDistribution) ,\
        (8, BinomialDistribution),\
        (9, BinomialDistributionEx),\
        (10, LogNormalDistribution) ,\
        (11, TruncatedNormalDistribution) ,\
        (12, AlphaDropOut),\
        (13, ExponentialDistribution),\
        (14, ExponentialDistributionInv)

#define PAIRWISE_BOOL_OPS \
        (0, EqualTo),\
        (1, GreaterThan),\
        (2, LessThan),\
        (3, Epsilon),\
        (4, GreaterThanOrEqual),\
        (5, LessThanOrEqual),\
        (6, NotEqualTo),\
        (7, And),\
        (8, Or),\
        (9, Xor) ,\
        (10, Not)

#define PAIRWISE_TRANSFORM_OPS \
        (0, Add),\
        (1, Copy),\
        (2, Divide),\
        (3, Multiply),\
        (4, Pow),\
        (5, ReverseSubtract),\
        (6, Subtract),\
        (7, MaxPairwise),\
        (8, MinPairwise),\
        (9, Copy2) ,\
        (10, Axpy),\
        (11, ReverseDivide),\
        (12, CompareAndSet),\
        (13, CompareAndReplace),\
        (14, Remainder),\
        (15, FMod),\
        (16, Atan2) ,\
        (17, TruncateDiv),\
        (18, FloorDiv), \
        (19, FloorMod) ,\
        (20, SquaredSubtract) ,\
        (21, ReverseMod),\
        (22, SafeDivide), \
        (23, Mod) ,\
        (24, RelativeError) ,\
        (25, BinaryRelativeError) ,\
        (26, BinaryMinimumAbsoluteRelativeError) ,\
        (27, LogicalOr) ,\
        (28, LogicalXor) ,\
        (29, LogicalNot) ,\
        (30, LogicalAnd) ,\
        (31, PowDerivative), \
        (32, LogPoisonLoss), \
        (33, LogPoisonLossFull)



#define INDEX_REDUCE_OPS \
        (0, IndexMax), \
        (1, IndexMin), \
        (2, IndexAbsoluteMax), \
	    (3, IndexAbsoluteMin) , \
	    (4, FirstIndex) , \
	    (5, LastIndex)



#endif //PROJECT_LEGACY_OPS_H
