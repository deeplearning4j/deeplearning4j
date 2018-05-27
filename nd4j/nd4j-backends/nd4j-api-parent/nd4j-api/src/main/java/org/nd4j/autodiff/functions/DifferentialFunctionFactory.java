package org.nd4j.autodiff.functions;

import lombok.Data;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.Min;
import org.nd4j.linalg.api.ops.impl.accum.distances.*;
import org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.layers.convolution.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.*;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterAdd;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterDiv;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterMul;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterSub;
import org.nd4j.linalg.api.ops.impl.shape.*;
import org.nd4j.linalg.api.ops.impl.shape.Stack;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.bp.*;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.nd4j.linalg.api.ops.random.impl.DropOut;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;

import java.lang.reflect.Method;
import java.util.*;

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


    public SDVariable var(String iName, SDVariable iX) {
        return SDVariable.builder()
                .shape(iX.getShape())
                .varName(iName)
                .sameDiff(sameDiff())
                .build();
    }


    public SDVariable zero(int[] shape) {
        return sameDiff.zero("one-" + UUID.randomUUID().toString(), shape);
    }

    public SDVariable zero(long[] shape) {
        return sameDiff.zero("one-" + UUID.randomUUID().toString(), shape);
    }

    public SDVariable zerosLike(SDVariable input) {
        return zerosLike(null, input);
    }

    public SDVariable zerosLike(String name, SDVariable input) {
        validateDifferentialFunctionsameDiff(input);
        return new ZerosLike(name, sameDiff(), input).outputVariables()[0];
    }


    public SDVariable one(int[] shape) {
        return one(ArrayUtil.toLongArray(shape));
    }

    public SDVariable one(long[] shape) {
        return sameDiff.one("one-" + UUID.randomUUID().toString(), shape);
    }

    public SDVariable onesLike(String name, SDVariable input) {
        validateDifferentialFunctionsameDiff(input);
        return new OnesLike(name, sameDiff(), input).outputVariables()[0];
    }

    /**
     * Local response normalization operation.
     *
     * @param inputs    the inputs to lrn
     * @param lrnConfig the configuration
     * @return
     */
    public SDVariable localResponseNormalization(SDVariable inputs, LocalResponseNormalizationConfig lrnConfig) {
        LocalResponseNormalization lrn = LocalResponseNormalization.builder()
                .inputFunctions(new SDVariable[]{inputs})
                .sameDiff(sameDiff())
                .config(lrnConfig)
                .build();

        return lrn.outputVariables()[0];
    }

    /**
     * Conv1d operation.
     *
     * @param inputs       the inputs to conv1d
     * @param conv1DConfig the configuration
     * @return
     */
    public SDVariable conv1d(SDVariable[] inputs, Conv1DConfig conv1DConfig) {
        Conv1D conv1D = Conv1D.builder()
                .inputFunctions(inputs)
                .sameDiff(sameDiff())
                .config(conv1DConfig)
                .build();

        return conv1D.outputVariables()[0];
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

        return conv2D.outputVariables()[0];
    }

    /**
     * Average pooling 2d operation.
     *
     * @param inputs          the inputs to pooling
     * @param pooling2DConfig the configuration
     * @return
     */
    public SDVariable avgPooling2d(SDVariable[] inputs, Pooling2DConfig pooling2DConfig) {
        AvgPooling2D avgPooling2D = AvgPooling2D.builder()
                .inputs(inputs)
                .sameDiff(sameDiff())
                .config(pooling2DConfig)
                .build();

        return avgPooling2D.outputVariables()[0];
    }

    /**
     * Max pooling 2d operation.
     *
     * @param inputs          the inputs to pooling
     * @param pooling2DConfig the configuration
     * @return
     */
    public SDVariable maxPooling2d(SDVariable[] inputs, Pooling2DConfig pooling2DConfig) {
        MaxPooling2D maxPooling2D = MaxPooling2D.builder()
                .inputs(inputs)
                .sameDiff(sameDiff())
                .config(pooling2DConfig)
                .build();

        return maxPooling2D.outputVariables()[0];
    }

    /**
     * Avg pooling 3d operation.
     *
     * @param inputs          the inputs to pooling
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable avgPooling3d(SDVariable[] inputs, Pooling3DConfig pooling3DConfig) {
        Pooling3D maxPooling3D = Pooling3D.builder()
                .inputs(inputs)
                .sameDiff(sameDiff())
                .pooling3DConfig(pooling3DConfig)
                .type(Pooling3D.Pooling3DType.AVG)
                .build();

        return maxPooling3D.outputVariables()[0];
    }


    /**
     * Max pooling 3d operation.
     *
     * @param inputs          the inputs to pooling
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable maxPooling3d(SDVariable[] inputs, Pooling3DConfig pooling3DConfig) {
        Pooling3D maxPooling3D = Pooling3D.builder()
                .inputs(inputs)
                .sameDiff(sameDiff())
                .pooling3DConfig(pooling3DConfig)
                .type(Pooling3D.Pooling3DType.MAX)
                .build();

        return maxPooling3D.outputVariables()[0];
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

        return sconv2D.outputVariables()[0];
    }


    /**
     * Depthwise Conv2d operation. This is just separable convolution with
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

        return depthWiseConv2D.outputVariables()[0];
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

        return deconv2D.outputVariables()[0];
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
                                double epsilon) {
        BatchNorm batchNorm = BatchNorm.builder()
                .inputFunctions(new SDVariable[]{input, mean, variance, gamma, beta})
                .applyGamma(applyGamma)
                .applyBeta(applyBeta)
                .epsilon(epsilon)
                .sameDiff(sameDiff())
                .build();

        val outputVars = batchNorm.outputVariables();
        return outputVars[0];
    }

    public SDVariable[] moments(SDVariable input, int... axes) {
        return new Moments(sameDiff(), input, axes).outputVariables();
    }

    public SDVariable[] normalizeMoments(SDVariable counts, SDVariable means, SDVariable variances, double shift) {
        return new NormalizeMoments(sameDiff(), counts, means, variances, shift).outputVariables();
    }


    public SDVariable tile(SDVariable iX, int[] repeat) {
        if (repeat == null) {
            throw new ND4JIllegalStateException("Repeat must not be null!");
        }
        return new Tile(sameDiff(), iX, repeat).outputVariables()[0];
    }


    public SDVariable dropout(SDVariable input, double p) {
        return new DropOut(sameDiff(), input, p).outputVariables()[0];
    }


    public SDVariable sum(SDVariable i_x,
                          int... dimensions) {
        return new Sum(sameDiff(), i_x, dimensions).outputVariables()[0];
    }


    public SDVariable prod(SDVariable i_x, int... dimensions) {
        return new Prod(sameDiff(), i_x, dimensions).outputVariables()[0];
    }


    public SDVariable mean(SDVariable i_x, int... dimensions) {
        return new Mean(sameDiff(), i_x, dimensions).outputVariables()[0];
    }


    public SDVariable std(SDVariable i_x,
                          boolean biasCorrected,
                          int... dimensions) {
        return new StandardDeviation(sameDiff(), i_x, dimensions, biasCorrected).outputVariables()[0];
    }


    public SDVariable variance(SDVariable i_x,
                               boolean biasCorrected,
                               int... dimensions) {
        return new Variance(sameDiff(), i_x, dimensions, biasCorrected).outputVariables()[0];
    }

    public SDVariable countNonZero(SDVariable input) {
        return new CountNonZero(sameDiff(), input).outputVariables()[0];
    }

    public SDVariable countZero(SDVariable input) {
        return new CountZero(sameDiff(), input).outputVariables()[0];
    }

    public SDVariable zeroFraction(SDVariable input) {
        return new ZeroFraction(sameDiff(), input).outputVariables()[0];
    }


    public SDVariable max(SDVariable i_x, int... dimensions) {
        return new Max(sameDiff(), i_x, dimensions).outputVariables()[0];
    }

    public SDVariable max(SDVariable first, SDVariable second) {
        return new org.nd4j.linalg.api.ops.impl.transforms.comparison.Max(sameDiff(), first, second)
                .outputVariables()[0];
    }


    public SDVariable min(SDVariable i_x, int... dimensions) {
        return new Min(sameDiff(), i_x, dimensions).outputVariables()[0];
    }

    public SDVariable min(SDVariable first, SDVariable second) {
        return new org.nd4j.linalg.api.ops.impl.transforms.comparison.Min(sameDiff(), first, second)
                .outputVariables()[0];
    }

    public SDVariable argmax(SDVariable in, int... dimensions) {
        return new IMax(sameDiff(), in, dimensions).outputVariables()[0];
    }

    public SDVariable argmin(SDVariable in, int... dimensions) {
        return new IMin(sameDiff(), in, dimensions).outputVariables()[0];
    }


    public SDVariable cumsum(SDVariable in, boolean exclusive, boolean reverse, int... dimensions) {
        return new CumSum(sameDiff(), in, exclusive, reverse, dimensions).outputVariables()[0];
    }

    public SDVariable cumprod(SDVariable in, boolean exclusive, boolean reverse, int... dimensions) {
        return new CumProd(sameDiff(), in, exclusive, reverse, dimensions).outputVariables()[0];
    }

    public SDVariable biasAdd(SDVariable input, SDVariable bias) {
        return new BiasAdd(sameDiff(), input, bias).outputVariables()[0];
    }

    public SDVariable norm1(SDVariable i_x, int... dimensions) {
        return new Norm1(sameDiff(), i_x, dimensions).outputVariables()[0];

    }


    public SDVariable norm2(SDVariable i_x, int... dimensions) {
        return new Norm2(sameDiff(), i_x, dimensions).outputVariables()[0];

    }


    public SDVariable normmax(SDVariable i_x, int... dimensions) {
        return new NormMax(sameDiff(), i_x, dimensions).outputVariables()[0];

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


    public SDVariable gradientBackwardsMarker(SDVariable iX) {
        return new GradientBackwardsMarker(sameDiff(), iX, sameDiff.scalar(iX.getVarName() + "-pairgrad", 1.0)).outputVariables()[0];
    }

    public SDVariable abs(SDVariable iX) {
        return new Abs(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable neg(SDVariable iX) {
        return new Negative(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable cos(SDVariable iX) {
        return new Cos(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable sin(SDVariable iX) {
        return new Sin(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable tan(SDVariable iX) {
        return new Tan(sameDiff(), iX, null).outputVariables()[0];

    }


    public SDVariable permute(SDVariable iX, int... dimensions) {
        return new Permute(sameDiff(), iX, dimensions).outputVariables()[0];
    }


    public SDVariable invertPermutation(SDVariable input, boolean inPlace) {
        return new InvertPermutation(sameDiff(), input, inPlace).outputVariables()[0];
    }

    public SDVariable transpose(SDVariable iX) {
        return new Transpose(sameDiff(), iX).outputVariables()[0];
    }


    public SDVariable acos(SDVariable iX) {
        return new ACos(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable asin(SDVariable iX) {
        return new ASin(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable atan(SDVariable iX) {
        return new ATan(sameDiff(), iX, null).outputVariables()[0];

    }

    public SDVariable atan2(SDVariable y, SDVariable x) {
        return new ATan2(sameDiff(), y, x).outputVariables()[0];
    }


    public SDVariable cosh(SDVariable iX) {
        return new Cosh(sameDiff(), iX, null).outputVariables()[0];

    }


    public SDVariable sinh(SDVariable iX) {
        return new Sinh(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable tanh(SDVariable iX) {
        return new Tanh(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable tanhDerivative(SDVariable iX, SDVariable wrt) {
        return new org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative(sameDiff(), iX, wrt).outputVariables()[0];
    }


    public SDVariable acosh(SDVariable iX) {
        return new ACosh(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable asinh(SDVariable iX) {
        return new ASinh(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable atanh(SDVariable iX) {
        return new ATanh(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable exp(SDVariable iX) {
        return new Exp(sameDiff(), iX, null).outputVariables()[0];
    }

    public SDVariable expm1(SDVariable iX) {
        return new Expm1(sameDiff(), iX, null).outputVariables()[0];
    }

    public SDVariable rsqrt(SDVariable iX) {
        return new RSqrt(sameDiff(), iX, null).outputVariables()[0];
    }

    public SDVariable log(SDVariable iX) {
        return new Log(sameDiff(), iX, null).outputVariables()[0];
    }

    public SDVariable log1p(SDVariable iX) {
        return new Log1p(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable isFinite(SDVariable ix) {
        return new IsFinite(sameDiff(), ix, null).outputVariables()[0];
    }


    public SDVariable isInfinite(SDVariable ix) {
        return new IsInf(sameDiff(), ix, null).outputVariables()[0];
    }

    public SDVariable isNaN(SDVariable ix) {
        return new IsNaN(sameDiff(), ix, null).outputVariables()[0];
    }

    public SDVariable round(SDVariable ix) {
        return new Round(sameDiff(), ix, null).outputVariables()[0];
    }

    public SDVariable or(SDVariable iX, SDVariable i_y) {
        return new Or(sameDiff(), iX, i_y).outputVariables()[0];
    }

    public SDVariable and(SDVariable ix, SDVariable iy) {
        return new And(sameDiff(), ix, iy).outputVariables()[0];
    }

    public SDVariable xor(SDVariable ix, SDVariable iy) {
        return new Xor(sameDiff(), ix, iy).outputVariables()[0];
    }


    public SDVariable eq(SDVariable iX, SDVariable i_y) {
        return new EqualTo(sameDiff(), new SDVariable[]{iX, i_y}, false).outputVariables()[0];
    }


    public SDVariable neq(SDVariable iX, double i_y) {
        return new ScalarNotEquals(sameDiff(), iX, i_y).outputVariables()[0];
    }


    public SDVariable neqi(SDVariable iX, double i_y) {
        return new ScalarNotEquals(sameDiff(), iX, i_y, true).outputVariables()[0];
    }


    public SDVariable neqi(SDVariable iX, SDVariable i_y) {
        return new NotEqualTo(sameDiff(), new SDVariable[]{iX, i_y}, true).outputVariables()[0];
    }

    public SDVariable neq(SDVariable iX, SDVariable i_y) {
        return new NotEqualTo(sameDiff(), new SDVariable[]{iX, i_y}, false).outputVariables()[0];
    }

    public SDVariable pow(SDVariable iX, double i_y) {
        return new Pow(sameDiff(), iX, false, i_y).outputVariables()[0];
    }


    public SDVariable sqrt(SDVariable iX) {
        return new Sqrt(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable square(SDVariable iX) {
        return new Pow(sameDiff(), iX, false, 2.0).outputVariables()[0];
    }


    public SDVariable cube(SDVariable iX) {
        return new Cube(sameDiff(), iX, null).outputVariables()[0];
    }

    public SDVariable cubeDerivative(SDVariable iX) {
        return new CubeDerivative(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable floor(SDVariable iX) {
        return new Floor(sameDiff(), iX, null).outputVariables()[0];
    }

    public SDVariable floorDiv(SDVariable x, SDVariable y) {
        return new FloorDivOp(sameDiff(), x, y).outputVariables()[0];
    }

    public List<SDVariable> floorDivBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new FloorDivBpOp(sameDiff(), x, y, grad).outputVariables());
    }

    public SDVariable floorMod(SDVariable x, SDVariable y) {
        return new FModOp(sameDiff(), x, y).outputVariables()[0];
    }

    public List<SDVariable> floorModBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new FloorModBpOp(sameDiff(), x, y, grad).outputVariables());
    }

    public SDVariable ceil(SDVariable x) {
        return new Ceil(sameDiff(), x).outputVariables()[0];
    }

    public SDVariable clipByValue(SDVariable x, double clipValueMin, double clipValueMax) {
        return new ClipByValue(sameDiff(), x, clipValueMin, clipValueMax).outputVariables()[0];
    }

    public SDVariable clipByNorm(SDVariable x, double clipValue) {
        return new ClipByNorm(sameDiff(), x, clipValue).outputVariables()[0];
    }


    public SDVariable relu(SDVariable iX, double cutoff) {
        return new RectifedLinear(sameDiff(), iX, false, cutoff).outputVariables()[0];
    }

    public SDVariable relu6(SDVariable iX, double cutoff) {
        return new Relu6(sameDiff(), iX, false, cutoff).outputVariables()[0];
    }


    public SDVariable softmax(SDVariable iX) {
        return new SoftMax(sameDiff(), new SDVariable[]{iX}).outputVariables()[0];
    }


    public SDVariable hardTanh(SDVariable iX) {
        return new HardTanh(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable hardTanhDerivative(SDVariable iX) {
        return new HardTanhDerivative(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable sigmoid(SDVariable iX) {
        return new Sigmoid(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable sigmoidDerivative(SDVariable iX, SDVariable wrt) {
        return new SigmoidDerivative(sameDiff(), iX, wrt).outputVariables()[0];
    }


    public SDVariable logSigmoid(SDVariable iX) {
        return new LogSigmoid(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable logSigmoidDerivative(SDVariable iX, SDVariable wrt) {
        return new LogSigmoidDerivative(sameDiff(), iX, wrt).outputVariables()[0];
    }

    public SDVariable powDerivative(SDVariable iX, double pow) {
        return new PowDerivative(sameDiff(), iX, false, pow).outputVariables()[0];
    }


    public SDVariable swish(SDVariable iX) {
        return new Swish(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable swishDerivative(SDVariable iX) {
        return new SwishDerivative(sameDiff(), iX, false).outputVariables()[0];
    }


    public SDVariable sign(SDVariable iX) {
        return new Sign(sameDiff(), iX, null).outputVariables()[0];
    }


    public SDVariable expandDims(SDVariable iX, int axis) {
        return new ExpandDims(sameDiff(), new SDVariable[]{iX}, axis).outputVariables()[0];
    }

    public SDVariable squeeze(SDVariable iX, int... axis) {
        return new Squeeze(sameDiff(), iX, axis).outputVariables()[0];
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred) {
        return new ConfusionMatrix(sameDiff(), labels, pred).outputVariables()[0];
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses) {
        return new ConfusionMatrix(sameDiff(), labels, pred, numClasses).outputVariables()[0];
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, SDVariable weights) {
        return new ConfusionMatrix(sameDiff(), labels, pred, weights).outputVariables()[0];
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights) {
        return new ConfusionMatrix(sameDiff(), labels, pred, numClasses, weights).outputVariables()[0];
    }

    public SDVariable broadcast(SDVariable iX, int... shape) {
        return broadcast(iX, ArrayUtil.toLongArray(shape));
    }

    public SDVariable broadcast(SDVariable iX, long... shape) {
        return new Broadcast(sameDiff(), iX, shape).outputVariables()[0];
    }

    public SDVariable onehot(SDVariable indices, int depth, int axis, double on, double off) {
        return new OneHot(sameDiff(), indices, depth, axis, on, off).outputVariables()[0];
    }

    public SDVariable onehot(SDVariable indices, int depth) {
        return new OneHot(sameDiff(), indices, depth).outputVariables()[0];
    }

    public SDVariable reciprocal(SDVariable a) {
        return new Reciprocal(sameDiff(), a, new Object[0]).outputVariables()[0];
    }


    public SDVariable repeat(SDVariable iX, int axis) {
        return new Repeat(sameDiff(), new SDVariable[]{iX}, axis).outputVariables()[0];

    }

    public SDVariable stack(SDVariable[] values, int axis) {
        return new Stack(sameDiff(), values, axis).outputVariables()[0];
    }

    public SDVariable parallel_stack(SDVariable[] values) {
        return new ParallelStack(sameDiff(), values).outputVariables()[0];
    }

    public SDVariable[] unstack(SDVariable value, int axis) {
        return new Unstack(sameDiff(), value, axis).outputVariables();
    }
    public SDVariable[] unstack(SDVariable value, int axis, int num) {
        return new Unstack(sameDiff(), value, axis, num).outputVariables();
    }

    public SDVariable assign(SDVariable x, SDVariable y) {
        return new Assign(sameDiff(), x, y).outputVariables()[0];
    }


    public SDVariable softsign(SDVariable iX) {
        return new SoftSign(sameDiff(), iX, null).outputVariables()[0];

    }


    public SDVariable softsignDerivative(SDVariable iX) {
        return new SoftSignDerivative(sameDiff(), iX, null).outputVariables()[0];

    }


    public SDVariable softplus(SDVariable iX) {
        return new SoftPlus(sameDiff(), iX, null).outputVariables()[0];

    }


    public SDVariable elu(SDVariable iX) {
        return new ELU(sameDiff(), iX, null).outputVariables()[0];

    }


    public SDVariable eluDerivative(SDVariable iX) {
        return new ELUDerivative(sameDiff(), iX, null).outputVariables()[0];

    }


    public SDVariable leakyRelu(SDVariable iX, double cutoff) {
        return new LeakyReLU(sameDiff(), iX, false, cutoff).outputVariables()[0];

    }

    public SDVariable leakyReluDerivative(SDVariable iX, double cutoff) {
        return new LeakyReLUDerivative(sameDiff(), iX, false, cutoff).outputVariables()[0];
    }


    public SDVariable reshape(SDVariable iX, int[] shape) {
        return new Reshape(sameDiff(), iX, ArrayUtil.toLongArray(shape)).outputVariables()[0];
    }

    public SDVariable reshape(SDVariable iX, long[] shape) {
        return new Reshape(sameDiff(), iX, shape).outputVariables()[0];
    }

    public SDVariable reverse(SDVariable x, int... dimensions) {
        return new Reverse(sameDiff(), x, dimensions).outputVariables()[0];
    }

    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths, int seq_dim, int batch_dim) {
        return new ReverseSequence(sameDiff(), x, seq_lengths, seq_dim, batch_dim).outputVariables()[0];
    }

    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths) {
        return new ReverseSequence(sameDiff(), x, seq_lengths).outputVariables()[0];
    }

    public SDVariable sequenceMask(SDVariable lengths, SDVariable maxLen) {
        return new SequenceMask(sameDiff(), lengths, maxLen).outputVariables()[0];
    }

    public SDVariable sequenceMask(SDVariable lengths, int maxLen) {
        return new SequenceMask(sameDiff(), lengths, maxLen).outputVariables()[0];
    }

    public SDVariable sequenceMask(SDVariable lengths) {
        return new SequenceMask(sameDiff(), lengths).outputVariables()[0];
    }

    public SDVariable rollAxis(SDVariable iX, int axis) {
        return new RollAxis(sameDiff(), iX, axis).outputVariables()[0];
    }

    public SDVariable concat(int dimension, SDVariable... inputs) {
        return new Concat(sameDiff(), dimension, inputs).outputVariables()[0];
    }

    public SDVariable fill(SDVariable shape, double value) {
        return new Fill(sameDiff(), shape, value).outputVariables()[0];
    }

    public SDVariable cosineSimilarity(SDVariable iX, SDVariable i_y, int... dimensions) {
        return new CosineSimilarity(sameDiff(), iX, i_y, dimensions).outputVariables()[0];
    }

    public SDVariable cosineDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return new CosineDistance(sameDiff(), ix, iy, dimensions).outputVariables()[0];
    }


    public SDVariable euclideanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return new EuclideanDistance(sameDiff(), iX, i_y, dimensions).outputVariables()[0];
    }


    public SDVariable manhattanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return new ManhattanDistance(sameDiff(), iX, i_y, dimensions).outputVariables()[0];
    }

    public SDVariable hammingDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return new HammingDistance(sameDiff(), ix, iy, dimensions).outputVariables()[0];
    }

    public SDVariable jaccardDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return new JaccardDistance(sameDiff(), ix, iy, dimensions).outputVariables()[0];
    }

    public SDVariable weightedCrossEntropyWithLogits(SDVariable targets, SDVariable inputs, SDVariable weights) {
        return new WeightedCrossEntropyLoss(sameDiff(), targets, inputs, weights).outputVariables()[0];
    }

    public SDVariable sigmoidCrossEntropyWithLogits(SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        return new SigmoidCrossEntropyLoss(sameDiff(), logits, weights, labels,
                reductionMode, labelSmoothing).outputVariables()[0];
    }

    public SDVariable softmaxCrossEntropyWithLogits(SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        return new SoftmaxCrossEntropyLoss(sameDiff(), logits, weights, labels,
                reductionMode, labelSmoothing).outputVariables()[0];
    }

    public SDVariable lossBinaryXENT(SDVariable iX,
                                     SDVariable i_y,
                                     int... dimensions) {
        throw new UnsupportedOperationException();
    }


    public SDVariable lossCosineSimilarity(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();
    }


    public SDVariable lossHinge(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossKLD(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossL1(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossL2(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMAE(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMAPE(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMSE(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMCXENT(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossMSLE(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossNegativeLogLikelihood(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossPoisson(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }


    public SDVariable lossSquaredHinge(SDVariable iX, SDVariable i_y, int... dimensions) {
        throw new UnsupportedOperationException();

    }

    public SDVariable xwPlusB(SDVariable input, SDVariable weights, SDVariable bias) {
        return new XwPlusB(sameDiff(), input, weights, bias).outputVariables()[0];
    }

    public SDVariable reluLayer(SDVariable input, SDVariable weights, SDVariable bias) {
        return new ReluLayer(sameDiff(), input, weights, bias).outputVariables()[0];
    }

    public SDVariable mmul(SDVariable x,
                           SDVariable y,
                           MMulTranspose mMulTranspose) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return new Mmul(sameDiff(), x, y, mMulTranspose).outputVariables()[0];
    }


    public SDVariable mmul(SDVariable x,
                           SDVariable y) {
        return mmul(x, y, MMulTranspose.allFalse());
    }


    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return new TensorMmul(sameDiff(), x, y, dimensions).outputVariables()[0];
    }


    public SDVariable softmaxDerivative(SDVariable functionInput, SDVariable wrt) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new SoftMaxDerivative(sameDiff(), functionInput, wrt).outputVariables()[0];
    }


    public SDVariable logSoftmax(SDVariable i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return new LogSoftMax(sameDiff(), i_v, null).outputVariables()[0];

    }


    public SDVariable logSoftmaxDerivative(SDVariable arg, SDVariable wrt) {
        validateDifferentialFunctionsameDiff(arg);
        return new LogSoftMaxDerivative(sameDiff(), arg, wrt).outputVariables()[0];
    }


    public SDVariable selu(SDVariable arg) {
        validateDifferentialFunctionsameDiff(arg);
        return new SELU(sameDiff(), arg, null).outputVariables()[0];
    }


    public SDVariable seluDerivative(SDVariable arg) {
        validateDifferentialFunctionsameDiff(arg);
        return new SELUDerivative(sameDiff(), arg, null).outputVariables()[0];
    }


    public SDVariable rsub(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RSubOp(sameDiff(), differentialFunction, i_v).outputVariables()[0];
    }

    public List<SDVariable> rsubBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new RSubBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable rdiv(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RDivOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariables()[0];
    }

    public List<SDVariable> rdivBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new RDivBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable rdivi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RDivOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariables()[0];

    }


    public SDVariable rsubi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new RSubOp(sameDiff(), differentialFunction, i_v).outputVariables()[0];

    }


    public SDVariable add(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new AddOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariables()[0];

    }

    public SDVariable mergeadd(SDVariable[] differentialFunctions) {
        for (SDVariable df : differentialFunctions)
            validateDifferentialFunctionsameDiff(df);
        return new MergeAddOp(sameDiff(), differentialFunctions, false).outputVariables()[0];
    }

    public SDVariable diag(SDVariable sdVariable) {
        validateDifferentialFunctionsameDiff(sdVariable);
        return new Diag(sameDiff(), new SDVariable[]{sdVariable}, false).outputVariables()[0];
    }

    public SDVariable diagPart(SDVariable sdVariable) {
        validateDifferentialFunctionsameDiff(sdVariable);
        return new DiagPart(sameDiff(), new SDVariable[]{sdVariable}, false).outputVariables()[0];
    }


    public SDVariable batchToSpace(SDVariable differentialFunction, int[] blocks, int[][] crops) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new BatchToSpace(sameDiff(), new SDVariable[]{differentialFunction}, blocks, crops, false)
                .outputVariables()[0];
    }

    public SDVariable spaceToBatch(SDVariable differentialFunction, int[] blocks, int[][] padding) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SpaceToBatch(sameDiff(), new SDVariable[]{differentialFunction}, blocks, padding, false)
                .outputVariables()[0];
    }

    public SDVariable depthToSpace(SDVariable differentialFunction, int blocksSize, String dataFormat) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DepthToSpace(sameDiff(), new SDVariable[]{differentialFunction}, blocksSize, dataFormat)
                .outputVariables()[0];
    }

    public SDVariable spaceToDepth(SDVariable differentialFunction, int blocksSize, String dataFormat) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SpaceToDepth(sameDiff(), new SDVariable[]{differentialFunction}, blocksSize, dataFormat)
                .outputVariables()[0];
    }

    public SDVariable[] dynamicPartition(SDVariable differentialFunction, SDVariable partitions, int numPartitions) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DynamicPartition(sameDiff(), differentialFunction, partitions, numPartitions)
                .outputVariables();
    }

    public SDVariable dynamicStitch(SDVariable[] indices, SDVariable[] differentialFunctions) {
        for (SDVariable df : differentialFunctions)
            validateDifferentialFunctionsameDiff(df);

        return new DynamicStitch(sameDiff(), indices, differentialFunctions)
                .outputVariables()[0];
    }

    public SDVariable dilation2D(SDVariable df, SDVariable weights, int[] strides,
                                 int[] rates, boolean isSameMode) {
        validateDifferentialFunctionsameDiff(df);
        return new Dilation2D(sameDiff(), new SDVariable[]{df, weights}, strides, rates, isSameMode, false)
                .outputVariables()[0];
    }

    public SDVariable shape(SDVariable df) {
        validateDifferentialFunctionsameDiff(df);
        return new org.nd4j.linalg.api.ops.impl.shape.Shape(sameDiff(), df, false).outputVariables()[0];
    }

    public SDVariable gather(SDVariable df, int axis, int[] broadcast) {
        validateDifferentialFunctionsameDiff(df);
        return new Gather(sameDiff(), df, axis, broadcast, false).outputVariables()[0];
    }

    public SDVariable gatherNd(SDVariable df, SDVariable indices) {
        validateDifferentialFunctionsameDiff(df);
        return new GatherNd(sameDiff(), df, indices, false).outputVariables()[0];
    }

    public SDVariable cross(SDVariable a, SDVariable b) {
        validateDifferentialFunctionsameDiff(a);
        return new Cross(sameDiff(), new SDVariable[]{a, b}).outputVariables()[0];
    }

    public SDVariable erf(SDVariable differentialFunction) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new Erf(sameDiff(), differentialFunction, false).outputVariables()[0];
    }

    public SDVariable erfc(SDVariable differentialFunction) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new Erfc(sameDiff(), differentialFunction, false).outputVariables()[0];
    }

    public SDVariable addi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new AddOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariables()[0];
    }

    public List<SDVariable> addBp(SDVariable x, SDVariable y, SDVariable grad) {
        SDVariable[] ret = new AddBpOp(sameDiff(), x, y, grad).outputVariables();
        return Arrays.asList(ret);
    }


    public SDVariable sub(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SubOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariables()[0];
    }

    public SDVariable squaredDifference(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SquaredDifferenceOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false)
                .outputVariables()[0];
    }


    public List<SDVariable> subBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new SubBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable subi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new SubOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariables()[0];

    }


    public SDVariable mul(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new MulOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariables()[0];
    }

    public List<SDVariable> mulBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new MulBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable muli(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new MulOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariables()[0];

    }


    public SDVariable div(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DivOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, false).outputVariables()[0];
    }

    public SDVariable truncatedDiv(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new TruncateDivOp(sameDiff(), differentialFunction, i_v, false).outputVariables()[0];
    }

    public List<SDVariable> divBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new DivBpOp(sameDiff(), x, y, grad).outputVariables());
    }


    public SDVariable divi(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new DivOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariables()[0];
    }


    public SDVariable rsub(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseSubtraction(sameDiff(), differentialFunction, i_v).outputVariables()[0];

    }


    public SDVariable rdiv(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseDivision(sameDiff(), differentialFunction, i_v).outputVariables()[0];

    }


    public SDVariable rdivi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseDivision(sameDiff(), differentialFunction, i_v, true).outputVariables()[0];
    }


    public SDVariable rsubi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarReverseSubtraction(sameDiff(), differentialFunction, i_v, true).outputVariables()[0];

    }


    public SDVariable add(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarAdd(sameDiff(), differentialFunction, i_v, false).outputVariables()[0];
    }


    public SDVariable addi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarAdd(sameDiff(), differentialFunction, i_v, true).outputVariables()[0];
    }


    public SDVariable sub(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarSubtraction(sameDiff(), differentialFunction, i_v).outputVariables()[0];
    }


    public SDVariable subi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarSubtraction(sameDiff(), differentialFunction, i_v, true).outputVariables()[0];

    }


    public SDVariable mul(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarMultiplication(sameDiff(), differentialFunction, i_v).outputVariables()[0];

    }


    public SDVariable muli(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarMultiplication(sameDiff(), differentialFunction, i_v, true).outputVariables()[0];

    }


    public SDVariable div(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarDivision(sameDiff(), differentialFunction, i_v).outputVariables()[0];
    }


    public SDVariable divi(SDVariable differentialFunction, double i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new ScalarDivision(sameDiff(), differentialFunction, i_v, true).outputVariables()[0];
    }


    public SDVariable gt(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThan(sameDiff(), new SDVariable[]{functionInput, functionInput1}, false).outputVariables()[0];
    }


    public SDVariable lt(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThan(sameDiff(), new SDVariable[]{functionInput, functionInput1}, false).outputVariables()[0];
    }


    public SDVariable gti(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThan(sameDiff(), new SDVariable[]{functionInput, functionInput1}, true).outputVariables()[0];
    }


    public SDVariable lti(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThan(sameDiff(), new SDVariable[]{functionInput, functionInput1}, true).outputVariables()[0];
    }


    public SDVariable gte(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThanOrEqual(sameDiff(), new SDVariable[]{functionInput, functionInput1}, false).outputVariables()[0];
    }


    public SDVariable lte(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThanOrEqual(sameDiff(), new SDVariable[]{functionInput, functionInput1}, false).outputVariables()[0];
    }


    public SDVariable gtei(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new GreaterThanOrEqual(sameDiff(), new SDVariable[]{functionInput, functionInput1}, true).outputVariables()[0];
    }


    public SDVariable ltOrEqi(SDVariable functionInput, SDVariable functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        validateDifferentialFunctionsameDiff(functionInput1);
        return new LessThanOrEqual(sameDiff(), new SDVariable[]{functionInput, functionInput1}, true).outputVariables()[0];
    }


    public SDVariable gt(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThan(sameDiff(), functionInput, functionInput1, false).outputVariables()[0];
    }


    public SDVariable lt(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThan(sameDiff(), functionInput, functionInput1, false).outputVariables()[0];
    }


    public SDVariable gti(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThan(sameDiff(), functionInput, functionInput1, true).outputVariables()[0];
    }


    public SDVariable lti(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThan(sameDiff(), functionInput, functionInput1, true).outputVariables()[0];
    }


    public SDVariable gte(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThanOrEqual(sameDiff(), functionInput, functionInput1, false).outputVariables()[0];
    }


    public SDVariable lte(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThanOrEqual(sameDiff(), functionInput, functionInput1, false).outputVariables()[0];
    }


    public SDVariable gtei(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarGreaterThanOrEqual(sameDiff(), functionInput, functionInput1, true).outputVariables()[0];
    }


    public SDVariable ltei(SDVariable functionInput, double functionInput1) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new ScalarLessThanOrEqual(sameDiff(), functionInput, functionInput1, true).outputVariables()[0];
    }


    public SDVariable eq(SDVariable iX, double i_y) {
        return new ScalarEquals(sameDiff(), iX, i_y).outputVariables()[0];
    }

    public SDVariable eqi(SDVariable iX, double i_y) {
        return new ScalarEquals(sameDiff(), iX, i_y, true).outputVariables()[0];
    }

    public SDVariable isNonDecreasing(SDVariable iX) {
        validateDifferentialFunctionsameDiff(iX);
        return new IsNonDecreasing(sameDiff(), new SDVariable[]{iX}, false).outputVariables()[0];
    }

    public SDVariable isStrictlyIncreasing(SDVariable iX) {
        validateDifferentialFunctionsameDiff(iX);
        return new IsStrictlyIncreasing(sameDiff(), new SDVariable[]{iX}, false).outputVariables()[0];
    }

    public SDVariable isNumericTensor(SDVariable iX) {
        validateDifferentialFunctionsameDiff(iX);
        return new IsNumericTensor(sameDiff(), new SDVariable[]{iX}, false).outputVariables()[0];
    }

    public SDVariable slice(SDVariable input, int[] begin, int[] size) {
        return new Slice(sameDiff(), input, begin, size).outputVariables()[0];
    }

    public SDVariable stridedSlice(SDVariable input, int[] begin, int[] end, int[] strides) {
        return new StridedSlice(sameDiff(), input, begin, end, strides).outputVariables()[0];
    }

    public SDVariable stridedSlice(SDVariable in, int[] begin, int[] end, int[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return new StridedSlice(sameDiff(), in, begin, end, strides, beginMask, endMask, ellipsisMask,
                newAxisMask, shrinkAxisMask).outputVariables()[0];
    }

    public SDVariable scatterAdd(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterAdd(sameDiff(), ref, indices, updates).outputVariables()[0];
    }

    public SDVariable scatterSub(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterSub(sameDiff(), ref, indices, updates).outputVariables()[0];
    }

    public SDVariable scatterMul(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterMul(sameDiff(), ref, indices, updates).outputVariables()[0];
    }

    public SDVariable scatterDiv(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterDiv(sameDiff(), ref, indices, updates).outputVariables()[0];
    }

    /**
     * @param func
     * @return
     */
    public long getInputLength(SDVariable func) {
        validateDifferentialFunctionsameDiff(func);
        long[] inputShape = func.arg().getShape();
        return ArrayUtil.prodLong(inputShape);
    }

    public long getReductionLength(DifferentialFunction func) {
        val inputShape = func.arg().getShape();
        if (Shape.isWholeArray(inputShape, func.getDimensions())) {
            return ArrayUtil.prod(inputShape);
        }
        int prod = 1;
        for (int i : func.getDimensions()) {
            prod *= inputShape[i];
        }
        return prod;
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
    public SDVariable doGradChoose(SDVariable func,
                                   SDVariable input) {
        validateDifferentialFunctionsameDiff(func);
        validateDifferentialFunctionsameDiff(input);

        SDVariable repeatedGrad = doRepeat(func, input);
        SDVariable resultRepeated = doRepeat(func.args()[0], input);
        SDVariable argMaxLocations = eq(input, resultRepeated);
        return div(mul(argMaxLocations, repeatedGrad), sum(argMaxLocations).outputVariables()[0]);
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


    public String toString() {
        return "DifferentialFunctionFactory{" +
                "methodNames=" + methodNames +
                '}';
    }


}
