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



#define BROADCAST_OPS \
       (0, Add), \
       (1, Subtract), \
       (2, Multiply), \
       (3, Divide), \
       (4, ReverseDivide), \
       (5, ReverseSubtract), \
       (6, Copy) ,\
       (7, EqualTo) ,\
       (8, GreaterThan) ,\
       (9, GreaterThanOrEqual) ,\
       (10, LessThan) ,\
       (11, LessThanOrEqual) ,\
       (12, NotEqualTo) ,\
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
       (24, And), \
       (25, Or) ,\
       (26, Atan2) ,\
       (27, LogicalOr) ,\
       (28, LogicalXor) ,\
       (29, LogicalNot) ,\
       (30, LogicalAnd)

// these ops return same data type as input
#define TRANSFORM_SAME_OPS \
        (0, Abs), \
        (1,Sign), \
        (2,Ones), \
        (3, Neg), \
        (4, Round), \
        (5,TimesOneMinus), \
        (6,Cube), \
        (7,OneMinus), \
        (8,Col2Im), \
        (9,Im2col),\
        (10,CubeDerivative) , \
        (11,Reciprocal), \
        (12,Square), \
        (13,RELU), \
        (14,Identity), \
        (15,Step), \
        (16, Ceiling), \
        (17, Floor), \
        (18,ClipByValue) ,\
        (19,Stabilize), \
        (20,StabilizeFP16) ,\
        (21,DropOut), \
        (22,Reverse)

// these ops return bool
#define TRANSFORM_BOOL_OPS \
        (0,IsMax), \
        (1,Not) ,\
        (2,IsInf), \
        (3,IsNan), \
        (4,IsFinite), \
        (5,MatchCondition)

// these ops return one of FLOAT data types
#define TRANSFORM_FLOAT_OPS \
        (0, Cosine), \
        (1, Exp), \
        (2, Log), \
        (3, SetRange), \
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
        (15,ELUDerivative), \
        (16,TanhDerivative), \
        (17,HardTanhDerivative), \
        (18,SigmoidDerivative), \
        (19,SoftSignDerivative), \
        (20,LeakyRELU), \
        (21,LeakyRELUDerivative), \
        (22,SoftMax), \
        (23,SoftMaxDerivative), \
        (24,LogSoftMax), \
        (25,SpecialDerivative), \
        (26,DropOutInverted), \
        (27,ReplaceNans) ,\
        (28,Histogram), \
        (29,HardSigmoid), \
        (30,HardSigmoidDerivative) ,\
        (31,RationalTanh) ,\
        (32,RationalTanhDerivative) ,\
        (33,LogX) ,\
        (34,RectifiedTanh) ,\
        (35,RectifiedTanhDerivative) ,\
        (36,Sinh) ,\
        (37,Cosh) ,\
        (38,Tan) ,\
        (39,TanDerivative) ,\
        (40,SELU) ,\
        (41,SELUDerivative) ,\
        (42,Pooling2D) ,\
        (43,Swish) ,\
        (44,SwishDerivative) ,\
        (45,RSqrt), \
        (46,Log1p), \
        (47,Erf), \
        (48,ACosh), \
        (49,ACoshDerivative) ,\
        (50,ASinh), \
        (51,ASinhDerivative) ,\
        (52,SinhDerivative), \
        (53,Rint), \
        (54,LogSigmoid), \
        (55,LogSigmoidDerivative) ,\
        (56,Erfc) ,\
        (57,Expm1), \
        (58,ATanh), \
        (59,RELU6)




#define SUMMARY_STATS_OPS \
        (0, SummaryStatsVariance), \
        (1, SummaryStatsStandardDeviation)




#define SCALAR_OPS \
        (0, Add),\
        (1, Subtract),\
        (2, Multiply),\
        (3, Divide),\
        (4, ReverseDivide),\
        (5, ReverseSubtract),\
        (6, MaxPairwise),\
        (7, LessThan),\
        (8, GreaterThan),\
        (9, EqualTo),\
        (10,LessThanOrEqual),\
        (11,NotEqualTo),\
        (12,MinPairwise),\
        (13,Copy),\
        (14,Mod),\
        (15,ReverseMod),\
        (16,GreaterThanOrEqual),\
        (17,Remainder),\
        (18,FMod) ,\
        (19, TruncateDiv) ,\
        (20, FloorDiv) ,\
        (21, FloorMod), \
        (22, SquaredSubtract),\
        (23, SafeDivide), \
        (24, AMaxPairwise), \
        (25, AMinPairwise), \
        (26, And), \
        (27, Or), \
        (28, Atan2) ,\
        (29, LogicalOr) ,\
        (30, LogicalXor) ,\
        (31, LogicalNot) ,\
        (32, LogicalAnd) ,\
        (33, Pow) ,\
        (34, PowDerivative) ,\
        (35, CompareAndSet) ,\
        (36, SXELogitsSmoother), \
        (37, LstmClip)





#define REDUCE3_OPS \
        (0, ManhattanDistance), \
        (1, EuclideanDistance), \
        (2, CosineSimilarity), \
        (3, Dot), \
        (4, EqualsWithEps) ,\
        (5, CosineDistance) ,\
        (6, JaccardDistance) ,\
        (7, SimpleHammingDistance)



#define REDUCE_OPS \
        (0, Mean), \
        (1, Sum), \
        (3, Max), \
        (4, Min), \
        (5, Norm1), \
        (6, Norm2), \
        (7, NormMax), \
        (8, Prod), \
        (9, StandardDeviation), \
        (10, Variance), \
        (11, ASum), \
        (12, MatchCondition) ,\
        (13, AMax) ,\
        (14, AMin) ,\
        (15, AMean) ,\
        (16, Entropy) ,\
        (17, LogEntropy) ,\
        (18, ShannonEntropy) ,\
        (19, LogSumExp) ,\
        (20, Any) ,\
        (21, All), \
        (22, CountNonZero), \
        (23, NormFrobenius), \
        (24, NormP), \
        (25, SquaredNorm), \
        (26, CountZero), \
        (27, IsFinite), \
        (28, IsInfOrNan), \
        (29, IsNan), \
        (30, IsInf)



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



#define PAIRWISE_TRANSFORM_OPS \
        (0, Add),\
        (1, Copy),\
        (2, Divide),\
        (3, EqualTo),\
        (4, GreaterThan),\
        (5, LessThan),\
        (6, Multiply),\
        (7, Pow),\
        (8, ReverseSubtract),\
        (9, Subtract),\
        (10,Epsilon),\
        (11,GreaterThanOrEqual),\
        (12,LessThanOrEqual),\
        (13,MaxPairwise),\
        (14,MinPairwise),\
        (15,NotEqualTo),\
        (16,Copy2),\
        (17,Axpy),\
        (18,ReverseDivide),\
        (45,CompareAndSet),\
        (46,CompareAndReplace),\
        (56,And),\
        (57,Or),\
        (58,Xor),\
        (59,Remainder),\
        (60,FMod),\
        (69,Atan2) ,\
        (19, TruncateDiv),\
        (20, FloorDiv), \
        (21, FloorMod) ,\
        (22, SquaredSubtract) ,\
        (23, ReverseMod),\
        (24, SafeDivide), \
        (25, Mod) ,\
        (26, RelativeError) ,\
        (27, BinaryRelativeError) ,\
        (28, BinaryMinimumAbsoluteRelativeError) ,\
        (29, LogicalOr) ,\
        (30, LogicalXor) ,\
        (31, LogicalNot) ,\
        (32, LogicalAnd) ,\
        (33, RELUDerivativeE) ,\
        (34, CUBEDerivativeE) ,\
        (35, ELUDerivativeE) ,\
        (36, HardSigmoidDerivativeE) ,\
        (37, HardTanhDerivativeE) ,\
        (38, LRELUDerivativeE) ,\
        (39, RationalTanhDerivativeE) ,\
        (40, RectifiedTanhDerivativeE) ,\
        (41, RELU6DerivativeE) ,\
        (42, SELUDerivativeE) ,\
        (43, SigmoidDerivativeE) ,\
        (44, SoftplusDerivativeE) ,\
        (47, SoftsignDerivativeE) ,\
        (48, TanhDerivativeE) ,\
        (49, SXELossWithLogits) ,\
        (50, ReduceNorm1E) ,\
        (92, PowDerivative), \
        (93, LogPoisonLoss), \
        (94, LogPoisonLossFull)



#define INDEX_REDUCE_OPS \
        (0, IndexMax), \
        (1, IndexMin), \
        (2, IndexAbsoluteMax), \
	    (3, IndexAbsoluteMin) , \
	    (4, FirstIndex) , \
	    (5, LastIndex)



#endif //PROJECT_LEGACY_OPS_H
