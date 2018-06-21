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
       (0, simdOps::Add), \
       (1, simdOps::Subtract), \
       (2, simdOps::Multiply), \
       (3, simdOps::Divide), \
       (4, simdOps::ReverseDivide), \
       (5, simdOps::ReverseSubtract), \
       (6, simdOps::Copy) ,\
       (7, simdOps::EqualTo) ,\
       (8, simdOps::GreaterThan) ,\
       (9, simdOps::GreaterThanOrEqual) ,\
       (10, simdOps::LessThan) ,\
       (11, simdOps::LessThanOrEqual) ,\
       (12, simdOps::NotEqualTo) ,\
       (13, simdOps::Min) ,\
       (14, simdOps::Max) ,\
       (15, simdOps::AMin) ,\
       (16, simdOps::AMax) ,\
       (17, simdOps::SquaredSubtract),\
       (18, simdOps::FloorMod),\
       (19, simdOps::FloorDiv),\
       (20, simdOps::ReverseMod),\
       (21, simdOps::SafeDivide),\
       (22, simdOps::Mod) ,\
       (23, simdOps::TruncateDiv), \
       (24, simdOps::And), \
       (25, simdOps::Or)




#define TRANSFORM_OPS \
	(0, simdOps::Abs), \
        (1, simdOps::Ceiling), \
        (2, simdOps::Cosine), \
        (3, simdOps::Exp), \
        (4, simdOps::Floor), \
        (5, simdOps::Log), \
        (6, simdOps::Neg), \
        (7, simdOps::Pow), \
        (8, simdOps::Round), \
        (9, simdOps::SetRange), \
        (10,simdOps::Sigmoid), \
        (11,simdOps::Sign), \
        (12,simdOps::Sin), \
        (13,simdOps::SoftPlus), \
        (14,simdOps::Sqrt), \
        (15,simdOps::Tanh), \
        (16,simdOps::ACos), \
        (17,simdOps::ASin), \
        (18,simdOps::ATan), \
        (19,simdOps::HardTanh), \
        (20,simdOps::SoftSign), \
        (21,simdOps::ELU), \
        (22,simdOps::ELUDerivative), \
        (23,simdOps::TanhDerivative), \
        (24,simdOps::TimesOneMinus), \
        (25,simdOps::HardTanhDerivative), \
        (26,simdOps::Ones), \
        (27,simdOps::Identity), \
        (28,simdOps::Stabilize), \
        (29,simdOps::SigmoidDerivative), \
        (30,simdOps::SoftSignDerivative), \
        (31,simdOps::LeakyRELU), \
        (32,simdOps::LeakyRELUDerivative), \
        (33,simdOps::RELU), \
        (34,simdOps::Step), \
        (35,simdOps::OneMinus), \
        (36,simdOps::Col2Im), \
        (37,simdOps::Im2col), \
        (38,simdOps::SoftMax), \
        (39,simdOps::SoftMaxDerivative), \
        (40,simdOps::LogSoftMax), \
        (41,simdOps::IsMax), \
        (42,simdOps::SpecialDerivative), \
        (43,simdOps::DropOut), \
        (44,simdOps::DropOutInverted), \
        (45,simdOps::CompareAndSet), \
        (46,simdOps::ReplaceNans) ,\
        (47,simdOps::StabilizeFP16) ,\
        (48,simdOps::Histogram), \
        (49,simdOps::Cube), \
        (50,simdOps::CubeDerivative) , \
        (51,simdOps::HardSigmoid), \
        (52,simdOps::HardSigmoidDerivative) ,\
        (53,simdOps::RationalTanh) ,\
        (54,simdOps::RationalTanhDerivative) ,\
        (55,simdOps::LogX) ,\
        (59,simdOps::Not) ,\
        (61,simdOps::RectifiedTanh) ,\
        (62,simdOps::RectifiedTanhDerivative) ,\
        (63,simdOps::Sinh) ,\
        (64,simdOps::Cosh) ,\
        (65,simdOps::Tan) ,\
        (66,simdOps::TanDerivative) ,\
        (67,simdOps::SELU) ,\
        (68,simdOps::SELUDerivative) ,\
        (70,simdOps::Reverse) ,\
        (71,simdOps::Pooling2D) ,\
        (72,simdOps::MatchCondition) ,\
        (73,simdOps::ClipByValue) ,\
        (74,simdOps::Swish) ,\
        (75,simdOps::SwishDerivative) ,\
        (76,simdOps::RSqrt), \
        (77,simdOps::Log1p), \
        (78,simdOps::Erf), \
        (79,simdOps::IsInf), \
        (80,simdOps::IsNan), \
        (81,simdOps::IsFinite), \
        (82,simdOps::ACosh), \
        (83,simdOps::ACoshDerivative) ,\
        (84,simdOps::ASinh), \
        (85,simdOps::ASinhDerivative) ,\
        (86,simdOps::SinhDerivative), \
        (87,simdOps::Rint), \
        (88,simdOps::LogSigmoid), \
        (89,simdOps::LogSigmoidDerivative) ,\
        (90,simdOps::Erfc) ,\
        (91,simdOps::Expm1), \
        (92, simdOps::PowDerivative), \
        (93,simdOps::ATanh), \
        (94,simdOps::Reciprocal), \
        (95,simdOps::Square), \
        (96,simdOps::RELU6)




#define SUMMARY_STATS_OPS \
        (0, simdOps::SummaryStatsVariance), \
        (1,     simdOps::SummaryStatsStandardDeviation)




#define SCALAR_OPS \
        (0, simdOps::Add),\
        (1, simdOps::Subtract),\
        (2, simdOps::Multiply),\
        (3, simdOps::Divide),\
        (4, simdOps::ReverseDivide),\
        (5, simdOps::ReverseSubtract),\
        (6, simdOps::Max),\
        (7, simdOps::LessThan),\
        (8, simdOps::GreaterThan),\
        (9, simdOps::EqualTo),\
        (10,simdOps::LessThanOrEqual),\
        (11,simdOps::NotEqualTo),\
        (12,simdOps::Min),\
        (13,simdOps::Copy),\
        (14,simdOps::Mod),\
        (15,simdOps::ReverseMod),\
        (16,simdOps::GreaterThanOrEqual),\
        (17,simdOps::Remainder),\
        (18,simdOps::FMod) ,\
        (19, simdOps::TruncateDiv) ,\
        (20, simdOps::FloorDiv) ,\
        (21, simdOps::FloorMod), \
        (22, simdOps::SquaredSubtract),\
        (23, simdOps::SafeDivide), \
        (24, simdOps::AMax), \
        (25, simdOps::AMin), \
        (26, simdOps::And), \
        (27, simdOps::Or)





#define REDUCE3_OPS \
        (0, simdOps::ManhattanDistance), \
        (1, simdOps::EuclideanDistance), \
        (2, simdOps::CosineSimilarity), \
        (3, simdOps::Dot), \
        (4, simdOps::EqualsWithEps) ,\
        (5, simdOps::CosineDistance) ,\
        (6, simdOps::JaccardDistance) ,\
        (7, simdOps::SimpleHammingDistance)



#define REDUCE_OPS \
        (0, simdOps::Mean), \
        (1, simdOps::Sum), \
        (3, simdOps::Max), \
        (4, simdOps::Min), \
        (5, simdOps::Norm1), \
        (6, simdOps::Norm2), \
        (7, simdOps::NormMax), \
        (8, simdOps::Prod), \
        (9, simdOps::StandardDeviation), \
        (10, simdOps::Variance), \
        (11, simdOps::ASum), \
        (12, simdOps::MatchCondition) ,\
        (13, simdOps::AMax) ,\
        (14, simdOps::AMin) ,\
        (15, simdOps::AMean) ,\
        (16, simdOps::Entropy) ,\
        (17, simdOps::LogEntropy) ,\
        (18, simdOps::ShannonEntropy) ,\
        (19, simdOps::LogSumExp) ,\
        (20, simdOps::Any) ,\
        (21, simdOps::All), \
        (22, simdOps::CountNonZero), \
        (23, simdOps::NormFrobenius), \
        (24, simdOps::NormP), \
        (25, simdOps::SquaredNorm), \
        (26, simdOps::CountZero), \
        (27, simdOps::IsFinite), \
        (28, simdOps::IsInfOrNan), \
        (29, simdOps::IsNan), \
        (30, simdOps::IsInf)



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
        (13, randomOps::ExponentialDistribution)



#define PAIRWISE_TRANSFORM_OPS \
        (0, simdOps::Add),\
        (1, simdOps::Copy),\
        (2, simdOps::Divide),\
        (3, simdOps::EqualTo),\
        (4, simdOps::GreaterThan),\
        (5, simdOps::LessThan),\
        (6, simdOps::Multiply),\
        (7, simdOps::Pow),\
        (8, simdOps::ReverseSubtract),\
        (9, simdOps::Subtract),\
        (10,simdOps::Epsilon),\
        (11,simdOps::GreaterThanOrEqual),\
        (12,simdOps::LessThanOrEqual),\
        (13,simdOps::Max),\
        (14,simdOps::Min),\
        (15,simdOps::NotEqualTo),\
        (16,simdOps::Copy2),\
        (17,simdOps::Axpy),\
        (18,simdOps::ReverseDivide),\
        (45,simdOps::CompareAndSet),\
        (46,simdOps::CompareAndReplace),\
        (56,simdOps::And),\
        (57,simdOps::Or),\
        (58,simdOps::Xor),\
        (59,simdOps::Remainder),\
        (60,simdOps::FMod),\
        (69,simdOps::Atan2) ,\
        (19, simdOps::TruncateDiv),\
        (20, simdOps::FloorDiv), \
        (21, simdOps::FloorMod) ,\
        (22, simdOps::SquaredSubtract) ,\
        (23, simdOps::ReverseMod),\
        (24, simdOps::SafeDivide), \
        (25, simdOps::Mod) ,\
        (26, simdOps::RelativeError) ,\
        (27, simdOps::BinaryRelativeError) ,\
        (28, simdOps::BinaryMinimumAbsoluteRelativeError) ,\
        (92, simdOps::PowDerivative), \
        (93, simdOps::LogPoisonLoss), \
        (94, simdOps::LogPoisonLossFull)



#define INDEX_REDUCE_OPS \
        (0, simdOps::IndexMax), \
        (1, simdOps::IndexMin), \
		(2, simdOps::IndexAbsoluteMax), \
		(3, simdOps::IndexAbsoluteMin) , \
		(4, simdOps::FirstIndex) , \
		(5, simdOps::LastIndex)



#endif //PROJECT_LEGACY_OPS_H
