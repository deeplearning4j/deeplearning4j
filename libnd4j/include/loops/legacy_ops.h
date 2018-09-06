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




#define TRANSFORM_OPS \
	(0, Abs), \
        (1, Ceiling), \
        (2, Cosine), \
        (3, Exp), \
        (4, Floor), \
        (5, Log), \
        (6, Neg), \
        (8, Round), \
        (9, SetRange), \
        (10,Sigmoid), \
        (11,Sign), \
        (12,Sin), \
        (13,SoftPlus), \
        (14,Sqrt), \
        (15,Tanh), \
        (16,ACos), \
        (17,ASin), \
        (18,ATan), \
        (19,HardTanh), \
        (20,SoftSign), \
        (21,ELU), \
        (22,ELUDerivative), \
        (23,TanhDerivative), \
        (24,TimesOneMinus), \
        (25,HardTanhDerivative), \
        (26,Ones), \
        (27,Identity), \
        (28,Stabilize), \
        (29,SigmoidDerivative), \
        (30,SoftSignDerivative), \
        (31,LeakyRELU), \
        (32,LeakyRELUDerivative), \
        (33,RELU), \
        (34,Step), \
        (35,OneMinus), \
        (36,Col2Im), \
        (37,Im2col), \
        (38,SoftMax), \
        (39,SoftMaxDerivative), \
        (40,LogSoftMax), \
        (41,IsMax), \
        (42,SpecialDerivative), \
        (43,DropOut), \
        (44,DropOutInverted), \
        (46,ReplaceNans) ,\
        (47,StabilizeFP16) ,\
        (48,Histogram), \
        (49,Cube), \
        (50,CubeDerivative) , \
        (51,HardSigmoid), \
        (52,HardSigmoidDerivative) ,\
        (53,RationalTanh) ,\
        (54,RationalTanhDerivative) ,\
        (55,LogX) ,\
        (59,Not) ,\
        (61,RectifiedTanh) ,\
        (62,RectifiedTanhDerivative) ,\
        (63,Sinh) ,\
        (64,Cosh) ,\
        (65,Tan) ,\
        (66,TanDerivative) ,\
        (67,SELU) ,\
        (68,SELUDerivative) ,\
        (70,Reverse) ,\
        (71,Pooling2D) ,\
        (72,MatchCondition) ,\
        (73,ClipByValue) ,\
        (74,Swish) ,\
        (75,SwishDerivative) ,\
        (76,RSqrt), \
        (77,Log1p), \
        (78,Erf), \
        (79,IsInf), \
        (80,IsNan), \
        (81,IsFinite), \
        (82,ACosh), \
        (83,ACoshDerivative) ,\
        (84,ASinh), \
        (85,ASinhDerivative) ,\
        (86,SinhDerivative), \
        (87,Rint), \
        (88,LogSigmoid), \
        (89,LogSigmoidDerivative) ,\
        (90,Erfc) ,\
        (91,Expm1), \
        (93,ATanh), \
        (94,Reciprocal), \
        (95,Square), \
        (96,RELU6)




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
        (35,CompareAndSet)





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
        (0, randomOps::UniformDistribution) ,\
        (1, randomOps::DropOut) ,\
        (2, randomOps::DropOutInverted) ,\
        (3, randomOps::ProbablisticMerge) ,\
        (4, randomOps::Linspace) ,\
        (5, randomOps::Choice) ,\
        (6, randomOps::GaussianDistribution) ,\
        (7, randomOps::BernoulliDistribution) ,\
        (8, randomOps::BinomialDistribution),\
        (9, randomOps::BinomialDistributionEx),\
        (10, randomOps::LogNormalDistribution) ,\
        (11, randomOps::TruncatedNormalDistribution) ,\
        (12, randomOps::AlphaDropOut),\
        (13, randomOps::ExponentialDistribution),\
        (14, randomOps::ExponentialDistributionInv)



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
