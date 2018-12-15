package org.nd4j.autodiff.samediff.serde;



import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.impl.*;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAMax;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAMin;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMax;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMin;
import org.nd4j.linalg.api.ops.impl.indexaccum.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Col2Im;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Im2col;
import org.nd4j.linalg.api.ops.impl.reduce.bool.All;
import org.nd4j.linalg.api.ops.impl.reduce.bool.Any;
import org.nd4j.linalg.api.ops.impl.reduce.bool.IsInf;
import org.nd4j.linalg.api.ops.impl.reduce.bool.IsNaN;
import org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp;
import org.nd4j.linalg.api.ops.impl.reduce.floating.*;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountZero;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.impl.reduce.same.AMax;
import org.nd4j.linalg.api.ops.impl.reduce.same.AMin;
import org.nd4j.linalg.api.ops.impl.reduce.same.*;
import org.nd4j.linalg.api.ops.impl.reduce.same.Max;
import org.nd4j.linalg.api.ops.impl.reduce.same.Min;
import org.nd4j.linalg.api.ops.impl.reduce3.*;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.*;
import org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.bool.IsFinite;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.api.ops.impl.transforms.custom.*;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Histogram;
import org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.*;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.BinaryMinimalRelativeError;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.BinaryRelativeError;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.RelativeError;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.Set;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.*;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.And;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Not;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Or;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Xor;
import org.nd4j.linalg.api.ops.impl.transforms.same.*;
import org.nd4j.linalg.api.ops.impl.transforms.strict.*;
import org.nd4j.linalg.api.ops.impl.transforms.strict.TanhDerivative;
import org.nd4j.linalg.api.ops.random.impl.*;

/**
 * This class maps legacy ops back to
 */
public class LegacyOpMapper {

    private LegacyOpMapper() {

    }

    public static Class<?> getLegacyOpClassForId(Op.Type opType, int opNum){
        switch (opType){
            case SCALAR:
                return scalarOpClass(opNum);
            case SCALAR_BOOL:
                return scalarBoolOpClass(opNum);
            case TRANSFORM_SAME:
                return transformSameOpClass(opNum);
            case TRANSFORM_STRICT:
                return transformStrictOpClass(opNum);
            case PAIRWISE:
                return pairwiseOpClass(opNum);
            case PAIRWISE_BOOL:
                return pairwiseBoolOpClass(opNum);
            case BROADCAST:
                return broadcastOpClass(opNum);
            case REDUCE_SAME:
                return reduceOpClass(opNum);
            case INDEXREDUCE:
                return indexReduceClass(opNum);
            case REDUCE3:
                return reduce3OpClass(opNum);
            case RANDOM:
                return randomOpClass(opNum);
            case AGGREGATION:
                return aggregateOpClass(opNum);
            case VARIANCE:  //Intentional fall-through
            case SUMMARYSTATS:
                return varianceOpClass(opNum);
            case REDUCE_FLOAT:
                return reduceOpClass(opNum);
            case TRANSFORM_BOOL:
                return transformBoolOpClass(opNum);
            case TRANSFORM_ANY:
                return transformAnyOpClass(opNum);
            case TRANSFORM_FLOAT:
                return transformFloatingOpClass(opNum);

            case SPECIAL:
            case GRID:
            case META:
            case CUSTOM:
            case GRADIENT:
            case SHAPE:
            case CONDITIONAL:
            case LOOP:
            case LOOP_COND:
            case IF:
            case RETURN:
            case ENTER:
            case EXIT:
            case NEXT_ITERATION:
            case MERGE:
            default:
                throw new UnsupportedOperationException("Unable to map op " + opNum + " of type " + opType);
        }
    }

    public static Class<? extends Aggregate> aggregateOpClass(int opNum) {
        switch (opNum) {
            case 0:
                return HierarchicSoftmax.class;
            case 1:
                return AggregateDot.class;
            case 2:
                return AggregateAxpy.class;
            case 3:
                return AggregateSkipGram.class;
            case 4:
                return AggregateCBOW.class;
            case 5:
                return AggregateGEMM.class;
            default:
                throw new UnsupportedOperationException("No known aggregate op for op number: " + opNum);
        }
    }

    public static Class<?> broadcastOpClass(int opNum) {
        switch (opNum) {
            case 0:
                return AddOp.class;
            case 1:
                return SubOp.class;
            case 2:
                return MulOp.class;
            case 3:
                return DivOp.class;
            case 4:
                return RDivOp.class;
            case 5:
                return RSubOp.class;
            case 6:
                return CopyOp.class;
            case 7:
                return EqualTo.class;
            case 8:
                return GreaterThan.class;
            case 9:
                return GreaterThanOrEqual.class;
            case 10:
                return LessThan.class;
            case 11:
                return LessThanOrEqual.class;
            case 12:
                return NotEqualTo.class;
            case 13:
                return BroadcastMin.class;
            case 14:
                return BroadcastMax.class;
            case 15:
                return BroadcastAMin.class;
            case 16:
                return BroadcastAMax.class;
            case 17:
                return SquaredDifferenceOp.class;
            case 18:
                return FloorModOp.class;
            case 19:
                return FloorDivOp.class;
            case 23:
                return TruncateDivOp.class;
            case 24:;
                return And.class;
            case 25:
                return Or.class;
            case 26:
                return OldAtan2Op.class;
            case 27:
                return LogicalOr.class;
            case 28:
                return LogicalXor.class;
            case 29:
                return Not.class;
            case 30:
                return LogicalAnd.class;
            case 20:    //RMod
            case 21:    //SafeDivide
            case 22:    //ModOp
            default:
                throw new UnsupportedOperationException("No known broadcast op for op number: " + opNum);
        }
    }

    public static Class<?> transformSameOpClass(int opNum){
        switch (opNum){
            case 0:
                return Abs.class;
            case 1:
                return Sign.class;
            case 3:
                return Negative.class;
            case 4:
                return Round.class;
            case 5:
                return TimesOneMinus.class;
            case 6:
                return Cube.class;
            case 7:
                return OneMinus.class;
            case 11:
                return Reciprocal.class;
            case 12:
                return Square.class;
            case 13:
                return CompareAndSet.class;
            case 17:
                return Ceil.class;
            case 18:
                return Floor.class;
            case 20:
                return OldReverse.class;
            default:
                throw new UnsupportedOperationException("No known transform same op for op number: " + opNum);
        }
    }

    public static Class<?> transformStrictOpClass(int opNum){
        switch (opNum){
            case 0:
                return Abs.class;
            case 2:
                return LogSoftMax.class;
            case 3:
                return ELUDerivative.class;
            case 4:
                return org.nd4j.linalg.api.ops.impl.transforms.strict.TanhDerivative.class;
            case 5:
                return HardTanhDerivative.class;
            case 6:
                return org.nd4j.linalg.api.ops.impl.transforms.strict.SigmoidDerivative.class;
            case 7:
                return SoftSignDerivative.class;
            case 8:
                return TanhDerivative.class;
            case 9:
                return SELUDerivative.class;
            case 10:
                return HardSigmoidDerivative.class;
            case 11:
                return RationalTanhDerivative.class;
            case 12:
                return RectifiedTanhDerivative.class;
            case 13:
                return SwishDerivative.class;
//            case 14:
//                return ACoshDerivative.class;
//            case 15:
//                return ASinhDerivative.class
//            case 16:
//                return SinhDerivative.class;
            case 17:
                return LogSigmoidDerivative.class;
            case 19:
                return Stabilize.class;
            case 21:
                return CubeDerivative.class;
            case 22:
                return Cos.class;
            case 23:
                return Exp.class;
            case 24:
                return Log.class;
            case 25:
                return SetRange.class;
            case 26:
                return Sigmoid.class;
            case 27:
                return Sin.class;
            case 28:
                return SoftPlus.class;
            case 29:
                return Tanh.class;
            case 30:
                return ACos.class;
            case 31:
                return ASin.class;
            case 32:
                return ATan.class;
            case 33:
                return HardTanh.class;
            case 34:
                return SoftSign.class;
            case 35:
                return ELU.class;
            case 36:
                return HardSigmoid.class;
            case 37:
                return RationalTanh.class;
            case 38:
                return RectifiedTanh.class;
            case 39:
                return Sinh.class;
            case 40:
                return Cosh.class;
            case 41:
                return Tan.class;
            case 42:
                return SELU.class;
            case 43:
                return Swish.class;
            case 44:
                return Log1p.class;
            case 45:
                return Erf.class;
            case 46:
                return ACosh.class;
            case 47:
                return ASinh.class;
            case 48:
                return Rint.class;
            case 49:
                return LogSigmoid.class;
            case 50:
                return Erfc.class;
            case 51:
                return Expm1.class;
            case 52:
                return ATanh.class;
            default:
                throw new UnsupportedOperationException("No known transform strict op for op number: " + opNum);
        }
    }

    public static Class<?> scalarOpClass(int opNum){
        switch (opNum){
            case 0:
                return ScalarAdd.class;
            case 1:
                return ScalarSubtraction.class;
            case 2:
                return ScalarMultiplication.class;
            case 3:
                return ScalarDivision.class;
            case 4:
                return ScalarReverseDivision.class;
            case 5:
                return ScalarReverseSubtraction.class;
            case 6:
                return ScalarMax.class;
            case 7:
                return ScalarLessThan.class;
            case 8:
                return ScalarGreaterThan.class;
            case 9:
                return ScalarEquals.class;
            case 10:
                return ScalarLessThanOrEqual.class;
            case 11:
                return ScalarNotEquals.class;
            case 13:
                return ScalarMin.class;
            case 14:
                return ScalarSet.class;
            case 16:
                return ScalarGreaterThanOrEqual.class;
            case 17:
                return ScalarRemainder.class;
            case 18:
                return ScalarFMod.class;
            case 31:
                return Pow.class;
            case 32:
                return PowDerivative.class;
            case 35:
                return LeakyReLU.class;
            case 37:
                return ReplaceNans.class;
            case 39:
                return RectifiedLinear.class;
            case 40:
                return Relu6.class;
            case 41:
                return Step.class;
            default:
                throw new UnsupportedOperationException("No known scalar op for op number: " + opNum);
        }
    }

    public static Class<?> scalarBoolOpClass(int opNum){
        switch (opNum){
            case 0:
                return ScalarEquals.class;
            case 1:
                return ScalarGreaterThan.class;
            case 2:
                return ScalarLessThan.class;
            case 3:
                return ScalarEps.class;
            case 4:
                return ScalarGreaterThanOrEqual.class;
            case 5:
                return ScalarLessThanOrEqual.class;
            case 6:
                return ScalarNotEquals.class;
            case 7:
                return ScalarAdd.class;
            case 8:
                return ScalarOr.class;
            case 9:
                return ScalarXor.class;
            case 10:
                return ScalarNot.class;
            default:
                throw new UnsupportedOperationException("No known scalar bool op for op number: " + opNum);
        }
    }

    public static Class<?> reduce3OpClass(int opNum){
        switch (opNum){
            case 0:
                return ManhattanDistance.class;
            case 1:
                return EuclideanDistance.class;
            case 2:
                return CosineSimilarity.class;
            case 3:
                return Dot.class;
            case 4:
                return EqualsWithEps.class;
            case 5:
                return CosineDistance.class;
            case 6:
                return JaccardDistance.class;
            case 7:
                return HammingDistance.class;
            default:
                throw new UnsupportedOperationException("No known reduce3 op for op number: " + opNum);
        }
    }

    public static Class<?> reduceOpClass(int opNum){
        switch (opNum) {
            case 0:
                return Mean.class;
            case 1:
                return Sum.class;
            case 3:
                return Max.class;
            case 4:
                return Min.class;
            case 5:
                return Norm1.class;
            case 6:
                return Norm2.class;
            case 7:
                return NormMax.class;
            case 8:
                return Prod.class;
            case 9:
                return StandardDeviation.class;
            case 10:
                return Variance.class;
            case 11:
                return ASum.class;
            case 12:
                return MatchCondition.class;
            case 13:
                return AMax.class;
            case 14:
                return AMin.class;
            case 15:
                return AMean.class;
            case 16:
                return Entropy.class;
            case 17:
                return LogEntropy.class;
            case 18:
                return ShannonEntropy.class;
            case 19:
                return LogSumExp.class;
            case 20:
                return Any.class;
            case 21:
                return All.class;
            case 22:
                return CountNonZero.class;
            case 25:
                return SquaredNorm.class;
            case 26:
                return CountZero.class;
            case 27:
                return IsFinite.class;
            case 29:
                return IsNaN.class;
            case 30:
                return IsInf.class;

            default:
                throw new UnsupportedOperationException("No known reduce op for op number: " + opNum);
//            case 23:
//                return NormFrobenius.class;
//            case 24:
//                return NormP.class;
//            case 28:
//                return IsInfOrNan.class;
        }
    }

    public static Class<?> randomOpClass(int opNum){
        switch (opNum){
            case 0:
                return UniformDistribution.class;
            case 1:
                return DropOut.class;
            case 2:
                return DropOutInverted.class;
            case 3:
                return ProbablisticMerge.class;
            case 4:
                return Linspace.class;
            case 5:
                return Choice.class;
            case 6:
                return GaussianDistribution.class;
            case 7:
                return BernoulliDistribution.class;
            case 8:
                return BinomialDistribution.class;
            case 9:
                return BinomialDistributionEx.class;
            case 10:
                return LogNormalDistribution.class;
            case 11:
                return TruncatedNormalDistribution.class;
            case 12:
                return AlphaDropOut.class;
            default:
                throw new UnsupportedOperationException("No known random op for op number: " + opNum);
//            case 13:
//                return ExponentialDistribution.class;
//            case 14:
//                return ExponentialDistributionInv.class;
        }
    }

    public static Class<?> pairwiseOpClass(int opNum){
        switch (opNum){
        case 0:
                return OldAddOp.class;
        case 1:
                return CopyOp.class;
        case 2:
                return OldDivOp.class;
        case 3:
                return OldEqualTo.class;
        case 4:
                return OldGreaterThan.class;
        case 5:
                return OldLessThan.class;
        case 6:
                return OldMulOp.class;
        case 7:
                return Pow.class;
        case 8:
                return RSubOp.class;
        case 9:
                return SubOp.class;
        case 10:
                return Eps.class;
        case 11:
                return OldGreaterThanOrEqual.class;
        case 12:
                return OldLessThanOrEqual.class;
        case 13:
                return OldMax.class;
        case 14:
                return OldMin.class;
        case 15:
                return OldNotEqualTo.class;
        case 16:
                return Set.class;
        case 17:
                return Axpy.class;
        case 18:
                return RDivOp.class;
        case 45:
                return CompareAndSet.class;
        case 46:
                return CompareAndReplace.class;
        case 56:
                return And.class;
        case 57:
                return Or.class;
        case 58:
                return Xor.class;
        case 59:
                return RemainderOp.class;
        case 60:
                return OldFModOp.class;
        case 69:
                return OldAtan2Op.class;
        case 20:
                return OldFloorDivOp.class;
        case 26:
                return RelativeError.class;
        case 27:
                return BinaryRelativeError.class;
        case 28:
                return BinaryMinimalRelativeError.class;
        case 92:
                return PowDerivative.class;
        default:
            throw new UnsupportedOperationException("No known pairwise op for op number: " + opNum);

//        case 19:
//            return TruncateDiv.class;
//            case 21:
//                return FloorMod.class;
//            case 22:
//                return SquaredSubtract.class;
//            case 23:
//                return ReverseMod.class;
//            case 24:
//                return SafeDivide.class;
//            case 25:
//                return Mod.class;
//            case 29:
//                return LogicalOr.class;
//            case 30:
//                return LogicalXor.class;
//            case 31:
//                return LogicalNot.class;
//            case 32:
//                return LogicalAnd.class;
//            case 93:
//                return LogPoisonLoss.class;
//            case 94:
//                return LogPoisonLossFull.class;
        }
    }

    public static Class<?> pairwiseBoolOpClass(int opNum){
        switch (opNum){
            case 7:
                return And.class;
            case 8:
                return Or.class;
            case 9:
                return Xor.class;
            default:
                throw new UnsupportedOperationException("No known pairwise bool op for op number: " + opNum);
        }
    }

    public static Class<?> indexReduceClass(int opNum){
        switch (opNum){
            case 0:
                return IMax.class;
            case 1:
                return IMin.class;
            case 2:
                return IAMax.class;
            case 3:
                return IAMin.class;
            case 4:
                return FirstIndex.class;
            case 5:
                return LastIndex.class;
            default:
                throw new UnsupportedOperationException("No known index reduce op for op number: " + opNum);
        }
    }

    public static Class<?> varianceOpClass(int opNum){
        switch (opNum){
            case 0:
                return Variance.class;
            case 1:
                return StandardDeviation.class;
            default:
                throw new UnsupportedOperationException("No known variance op for op number: " + opNum);
        }
    }

    public static Class<?> transformBoolOpClass(int opNum){
        switch (opNum){
            case 1:
                return org.nd4j.linalg.api.ops.impl.transforms.bool.IsInf.class;
            case 2:
                return org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN.class;
            case 3:
                return org.nd4j.linalg.api.ops.impl.transforms.bool.IsFinite.class;
            case 5:
                return org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform.class;
            case 7:
                return org.nd4j.linalg.api.ops.impl.transforms.bool.BooleanNot.class;
            default:
                throw new UnsupportedOperationException("No known transform bool op for op number: " + opNum);
        }
    }

    public static Class<?> transformAnyOpClass(int opNum){
        switch (opNum){
            case 0:
                return Assign.class;
            case 1:
                return IsMax.class;
            default:
                throw new UnsupportedOperationException("No known transform any op for op number: " + opNum);
        }
    }

    public static Class<?> transformFloatingOpClass(int opNum){
        switch (opNum){
            case 0:
                return Histogram.class;
            case 1:
                return Sqrt.class;
            case 3:
                return RSqrt.class;
            default:
                throw new UnsupportedOperationException("No known transform floating op for op number: " + opNum);
        }
    }

}
