package org.nd4j.autodiff.functions;

import lombok.Data;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.NoOp;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.Min;
import org.nd4j.linalg.api.ops.impl.accum.bp.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.*;
import org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd;
import org.nd4j.linalg.api.ops.impl.broadcast.BiasAddGrad;
import org.nd4j.linalg.api.ops.impl.indexaccum.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.*;
import org.nd4j.linalg.api.ops.impl.scatter.*;
import org.nd4j.linalg.api.ops.impl.shape.*;
import org.nd4j.linalg.api.ops.impl.shape.Stack;
import org.nd4j.linalg.api.ops.impl.shape.bp.SliceBp;
import org.nd4j.linalg.api.ops.impl.shape.bp.StridedSliceBp;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.bp.*;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.temp.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.random.custom.DistributionUniform;
import org.nd4j.linalg.api.ops.random.custom.RandomBernoulli;
import org.nd4j.linalg.api.ops.random.custom.RandomExponential;
import org.nd4j.linalg.api.ops.random.custom.RandomNormal;
import org.nd4j.linalg.api.ops.random.impl.*;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.indexing.conditions.Condition;
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

    public ExternalErrorsFunction externalErrors(SDVariable... inputs){
        return externalErrors(null, inputs);
    }

    public ExternalErrorsFunction externalErrors(Map<String,INDArray> externalGradients, SDVariable... inputs){
        Preconditions.checkArgument(inputs != null && inputs.length > 0, "Require at least one SDVariable to" +
                " be specified when using external errors: got %s", inputs);
        ExternalErrorsFunction fn = new ExternalErrorsFunction(sameDiff(), Arrays.asList(inputs), externalGradients);
        fn.outputVariable();
        return fn;
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
        return new ZerosLike(name, sameDiff(), input).outputVariable();
    }


    public SDVariable one(int[] shape) {
        return one(ArrayUtil.toLongArray(shape));
    }

    public SDVariable one(long[] shape) {
        return sameDiff.one("one-" + UUID.randomUUID().toString(), shape);
    }

    public SDVariable onesLike(String name, SDVariable input) {
        validateDifferentialFunctionsameDiff(input);
        return new OnesLike(name, sameDiff(), input).outputVariable();
    }

    public SDVariable constant(SDVariable input, long... shape){
        return new Constant(sameDiff(), input, (shape != null && shape.length > 0 ? null : shape)).outputVariable();
    }

    public SDVariable linspace(double lower, double upper, long count){
        return new Linspace(sameDiff(), lower, upper, count).outputVariable();
    }

    public SDVariable range(double from, double to, double step){
        return new Range(sameDiff(), from, to, step).outputVariable();
    }

    public SDVariable[] meshgrid(boolean cartesian, SDVariable... inputs){
        return new MeshGrid(sameDiff(), cartesian, inputs).outputVariables();
    }

    public SDVariable randomUniform(double min, double max, SDVariable shape){
        return new DistributionUniform(sameDiff(), shape, min, max).outputVariable();
    }

    public SDVariable randomUniform(double min, double max, long... shape){
        return new UniformDistribution(sameDiff(), min, max, shape).outputVariable();
    }

    public SDVariable randomNormal(double mean, double std, SDVariable shape){
        return new RandomNormal(sameDiff(), shape, mean, std).outputVariable();
    }

    public SDVariable randomNormal(double mean, double std, long... shape){
        return new GaussianDistribution(sameDiff(), mean, std, shape).outputVariable();
    }

    public SDVariable randomBernoulli(double p, SDVariable shape){
        return new RandomBernoulli(sameDiff(), shape, p).outputVariable();
    }

    public SDVariable randomBernoulli(double p, long... shape){
        return new BernoulliDistribution(sameDiff(), p, shape).outputVariable();
    }

    public SDVariable randomBinomial(int nTrials, double p, long... shape){
        return new BinomialDistribution(sameDiff(), nTrials, p, shape).outputVariable();
    }

    public SDVariable randomLogNormal(double mean, double stdev, long... shape){
        return new LogNormalDistribution(sameDiff(), mean, stdev, shape).outputVariable();
    }

    public SDVariable randomNormalTruncated(double mean, double stdev, long... shape){
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


    /**
     * Local response normalization operation.
     *
     * @param input    the inputs to lrn
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
     * @param input       the inputs to conv1d
     * @param weights     conv1d weights
     * @param conv1DConfig the configuration
     * @return
     */
    public SDVariable conv1d(SDVariable input, SDVariable weights, Conv1DConfig conv1DConfig) {
        Conv1D conv1D = Conv1D.builder()
                .inputFunctions(new SDVariable[] {input, weights})
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

    public SDVariable upsampling2d(SDVariable input, boolean nchw, int scaleH, int scaleW){
        return new Upsampling2d(sameDiff(), input, nchw, scaleH, scaleW).outputVariable();
    }

    public SDVariable upsampling2dBp(SDVariable input, SDVariable gradient, boolean nchw, int scaleH, int scaleW){
        return new Upsampling2dDerivative(sameDiff(), input, gradient, nchw, scaleH, scaleW).outputVariable();
    }



    /**
     * Average pooling 2d operation.
     *
     * @param input          the inputs to pooling
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
     * @param input          the inputs to pooling
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
     * @param input          the inputs to pooling
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable avgPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        pooling3DConfig.setType(Pooling3D.Pooling3DType.AVG);
        return pooling3d(input, pooling3DConfig);
    }


    /**
     * Max pooling 3d operation.
     *
     * @param input          the inputs to pooling
     * @param pooling3DConfig the configuration
     * @return
     */
    public SDVariable maxPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        pooling3DConfig.setType(Pooling3D.Pooling3DType.MAX);
        return pooling3d(input, pooling3DConfig);
    }
    public SDVariable pooling3d(SDVariable input, Pooling3DConfig pooling3DConfig){
        Pooling3D pool3d = Pooling3D.builder()
                .inputs(new SDVariable[]{input})
                .sameDiff(sameDiff())
                .pooling3DConfig(pooling3DConfig)
                .type(pooling3DConfig.getType())
                .build();
        return pool3d.outputVariable();
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

    public SDVariable im2Col(SDVariable input, Conv2DConfig config){
        return new Im2col(sameDiff(), input, config).outputVariable();
    }

    public SDVariable im2ColBp(SDVariable im2colInput, SDVariable gradientAtOutput, Conv2DConfig config){
        return new Im2colBp(sameDiff(), im2colInput, gradientAtOutput, config).outputVariable();
    }

    public SDVariable col2Im(SDVariable input, Conv2DConfig config){
        return new Col2Im(sameDiff(), input, config).outputVariable();
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
        return new Tile(sameDiff(), iX, repeat).outputVariable();
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

    public SDVariable mean(SDVariable in, boolean keepDims, int... dimensions){
        return new Mean(sameDiff(), in, keepDims, dimensions).outputVariable();
    }

    public SDVariable meanBp(SDVariable in, SDVariable grad, boolean keepDims, int... dimensions){
        return new MeanBp(sameDiff(), in, grad, keepDims, dimensions).outputVariable();
    }


    public SDVariable std(SDVariable i_x, boolean biasCorrected, boolean keepDims, int... dimensions) {
        return new StandardDeviation(sameDiff(), i_x, biasCorrected, keepDims, dimensions).outputVariable();
    }

    public SDVariable stdBp(SDVariable stdInput, SDVariable gradient, boolean biasCorrected, boolean keepDims, int... dimensions){
        return new StandardDeviationBp(sameDiff(), stdInput, gradient, biasCorrected, keepDims, dimensions).outputVariable();
    }


    public SDVariable variance(SDVariable i_x, boolean biasCorrected, boolean keepDims, int... dimensions) {
        return new Variance(sameDiff(), i_x, biasCorrected, keepDims, dimensions).outputVariable();
    }

    public SDVariable varianceBp(SDVariable stdInput, SDVariable gradient, boolean biasCorrected, boolean keepDims, int... dimensions){
        return new VarianceBp(sameDiff(), stdInput, gradient, biasCorrected, keepDims, dimensions).outputVariable();
    }

    public SDVariable squaredNorm(SDVariable input, boolean keepDims, int... dimensions){
        return new SquaredNorm(sameDiff(), input, keepDims, dimensions).outputVariable();
    }

    public SDVariable squaredNormBp(SDVariable preReduceInput, SDVariable gradient, boolean keepDims, int... dimensions){
        return new SquaredNormBp(sameDiff(), preReduceInput, gradient, keepDims, dimensions).outputVariable();
    }

    public SDVariable entropy(SDVariable in, int... dimensions){
        return new Entropy(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable logEntropy(SDVariable in, int... dimensions){
        return new LogEntropy(sameDiff(), in, dimensions).outputVariable();
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

    public SDVariable scalarMax(SDVariable in, Number num){
        return new ScalarMax(sameDiff(), in, num).outputVariable();
    }
    
    public SDVariable scalarMin(SDVariable in, Number num){
        return new ScalarMin(sameDiff(), in, num).outputVariable();
    }

    public SDVariable scalarSet(SDVariable in, Number num){
        return new ScalarSet(sameDiff(), in, num).outputVariable();
    }

    public SDVariable scalarFloorMod(SDVariable in, Number num){
        return new ScalarFMod(sameDiff(), in, num).outputVariable();
    }

    public SDVariable max(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new Max(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable max(SDVariable first, SDVariable second) {
        return new org.nd4j.linalg.api.ops.impl.transforms.comparison.Max(sameDiff(), first, second)
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
        return new org.nd4j.linalg.api.ops.impl.transforms.comparison.Min(sameDiff(), first, second)
                .outputVariable();
    }

    public SDVariable amax(SDVariable in, int... dimensions){
        return new AMax(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable amin(SDVariable in, int... dimensions){
        return new AMin(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable amean(SDVariable in, int... dimensions){
        return new AMean(sameDiff(), in, dimensions).outputVariable();
    }

    public SDVariable asum(SDVariable in, int... dimensions){
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

    public SDVariable firstIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        return new FirstIndex(sameDiff(), in, condition, keepDims, dimensions).outputVariable();
    }

    public SDVariable lastIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        return new LastIndex(sameDiff(), in, condition, keepDims, dimensions).outputVariable();
    }

    /**
     * Returns a count of the number of elements that satisfy the condition
     * @param in        Input
     * @param condition Condition
     * @return          Number of elements that the condition is satisfied for
     */
    public SDVariable matchConditionCount(SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        return new MatchCondition(sameDiff(), in, condition, keepDims, dimensions).outputVariable();
    }

    /**
     * Returns a boolean mask of equal shape to the input, where the condition is satisfied
     * @param in        Input
     * @param condition Condition
     * @return          Boolean mask
     */
    public SDVariable matchCondition(SDVariable in, Condition condition){
        return new MatchConditionTransform(sameDiff(), in, condition).outputVariable();
    }

    public SDVariable cumsum(SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        return new CumSum(sameDiff(), in, axis, exclusive, reverse).outputVariable();
    }

    public SDVariable cumsumBp(SDVariable in, SDVariable axis, SDVariable grad, boolean exclusive, boolean reverse) {
        return new CumSumBp(sameDiff(), in, axis, grad, exclusive, reverse).outputVariable();
    }

    public SDVariable cumprod(SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        return new CumProd(sameDiff(), in, axis, exclusive, reverse).outputVariable();
    }

    public SDVariable cumprodBp(SDVariable in, SDVariable axis, SDVariable grad, boolean exclusive, boolean reverse) {
        return new CumProdBp(sameDiff(), in, axis, grad, exclusive, reverse).outputVariable();
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

    public SDVariable norm1Bp(SDVariable preReduceIn, SDVariable grad, boolean keepDims, int... dimensions){
        return new Norm1Bp(sameDiff(), preReduceIn, grad, keepDims, dimensions).outputVariable();
    }

    public SDVariable norm2(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new Norm2(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable norm2Bp(SDVariable preReduceIn, SDVariable grad, boolean keepDims, int... dimensions){
        return new Norm2Bp(sameDiff(), preReduceIn, grad, keepDims, dimensions).outputVariable();
    }

    public SDVariable normmax(SDVariable i_x, boolean keepDims, int... dimensions) {
        return new NormMax(sameDiff(), i_x, keepDims, dimensions).outputVariable();
    }

    public SDVariable normmaxBp(SDVariable preReduceIn, SDVariable grad, boolean keepDims, int... dimensions){
        return new NormMaxBp(sameDiff(), preReduceIn, grad, keepDims, dimensions).outputVariable();
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
        return new GradientBackwardsMarker(sameDiff(), iX, sameDiff.scalar(iX.getVarName() + "-pairgrad", 1.0)).outputVariable();
    }

    public SDVariable abs(SDVariable iX) {
        return new Abs(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable neg(SDVariable iX) {
        return new Negative(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable cos(SDVariable iX) {
        return new Cos(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable sin(SDVariable iX) {
        return new Sin(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable tan(SDVariable iX) {
        return new Tan(sameDiff(), iX, false).outputVariable();

    }


    public SDVariable permute(SDVariable iX, int... dimensions) {
        return new Permute(sameDiff(), iX, dimensions).outputVariable();
    }

    public SDVariable noop(SDVariable input){
        return new NoOp(sameDiff(), input).outputVariable();
    }

    public SDVariable identity(SDVariable input){
        return new Identity(sameDiff(), input).outputVariable();
    }

    public SDVariable all(SDVariable input, int... dimensions){
        return new All(sameDiff(), input, dimensions).outputVariable();
    }

    public SDVariable any(SDVariable input, int... dimensions){
        return new Any(sameDiff(), input, dimensions).outputVariable();
    }

    public SDVariable invertPermutation(SDVariable input, boolean inPlace) {
        return new InvertPermutation(sameDiff(), input, inPlace).outputVariable();
    }

    public SDVariable transpose(SDVariable iX) {
        return new Transpose(sameDiff(), iX).outputVariable();
    }


    public SDVariable acos(SDVariable iX) {
        return new ACos(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable asin(SDVariable iX) {
        return new ASin(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable atan(SDVariable iX) {
        return new ATan(sameDiff(), iX, null).outputVariable();

    }

    public SDVariable atan2(SDVariable y, SDVariable x) {
        return new ATan2(sameDiff(), y, x).outputVariable();
    }


    public SDVariable cosh(SDVariable iX) {
        return new Cosh(sameDiff(), iX, null).outputVariable();

    }


    public SDVariable sinh(SDVariable iX) {
        return new Sinh(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable tanh(SDVariable iX) {
        return new Tanh(sameDiff(), iX, null).outputVariable();
    }

    public SDVariable tanhRational(SDVariable in){
        return new RationalTanh(sameDiff(), in, false).outputVariable();
    }

    public SDVariable tanhRectified(SDVariable in){
        return new RectifiedTanh(sameDiff(), in, false).outputVariable();
    }

    public SDVariable tanhDerivative(SDVariable iX, SDVariable wrt) {
        return new org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative(sameDiff(), iX, wrt).outputVariable();
    }

    public SDVariable tanhRationalDerivative(SDVariable in){
        return new RationalTanhDerivative(sameDiff(), in, false).outputVariable();
    }

    public SDVariable tanhRectifiedDerivative(SDVariable in){
        return new RectifiedTanhDerivative(sameDiff(), in, false).outputVariable();
    }

    public SDVariable step(SDVariable in, double cutoff){
        return new Step(sameDiff(), in, false, cutoff).outputVariable();
    }


    public SDVariable acosh(SDVariable iX) {
        return new ACosh(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable asinh(SDVariable iX) {
        return new ASinh(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable atanh(SDVariable iX) {
        return new ATanh(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable exp(SDVariable iX) {
        return new Exp(sameDiff(), iX, false).outputVariable();
    }

    public SDVariable expm1(SDVariable iX) {
        return new Expm1(sameDiff(), iX, null).outputVariable();
    }

    public SDVariable rsqrt(SDVariable iX) {
        return new RSqrt(sameDiff(), iX, null).outputVariable();
    }

    public SDVariable log(SDVariable iX) {
        return new Log(sameDiff(), iX, null).outputVariable();
    }

    public SDVariable log(SDVariable in, double base){
        return new LogX(sameDiff(), in, base).outputVariable();
    }

    public SDVariable log1p(SDVariable iX) {
        return new Log1p(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable isFinite(SDVariable ix) {
        return new IsFinite(sameDiff(), ix, null).outputVariable();
    }

    public SDVariable isInfinite(SDVariable ix) {
        return new IsInf(sameDiff(), ix, null).outputVariable();
    }

    public SDVariable isNaN(SDVariable ix) {
        return new IsNaN(sameDiff(), ix, null).outputVariable();
    }

    public SDVariable isMax(SDVariable ix){
        return new IsMax(sameDiff(), ix, false).outputVariable();
    }

    public SDVariable replaceWhere(SDVariable to, SDVariable from, Condition condition){
        return new CompareAndReplace(sameDiff(), to, from, condition).outputVariable();
    }

    public SDVariable replaceWhere(SDVariable to, Number set, Condition condition){
        return new CompareAndSet(sameDiff(), to, set, condition).outputVariable();
    }

    public SDVariable round(SDVariable ix) {
        return new Round(sameDiff(), ix, null).outputVariable();
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


    public SDVariable sqrt(SDVariable iX) {
        return new Sqrt(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable square(SDVariable iX) {
        return new Square(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable cube(SDVariable iX) {
        return new Cube(sameDiff(), iX, null).outputVariable();
    }

    public SDVariable cubeDerivative(SDVariable iX) {
        return new CubeDerivative(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable floor(SDVariable iX) {
        return new Floor(sameDiff(), iX, null).outputVariable();
    }

    public SDVariable floorDiv(SDVariable x, SDVariable y) {
        return new FloorDivOp(sameDiff(), x, y).outputVariable();
    }

    public List<SDVariable> floorDivBp(SDVariable x, SDVariable y, SDVariable grad) {
        return Arrays.asList(new FloorDivBpOp(sameDiff(), x, y, grad).outputVariables());
    }

    public SDVariable floorMod(SDVariable x, SDVariable y) {
        return new FModOp(sameDiff(), x, y).outputVariable();
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
        return new RectifedLinear(sameDiff(), iX, false, cutoff).outputVariable();
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


    public SDVariable hardTanh(SDVariable iX) {
        return new HardTanh(sameDiff(), iX, null).outputVariable();
    }

    public SDVariable hardTanhDerivative(SDVariable iX) {
        return new HardTanhDerivative(sameDiff(), iX, null).outputVariable();
    }

    public SDVariable hardSigmoid(SDVariable in){
        return new HardSigmoid(sameDiff(), in, false).outputVariable();
    }


    public SDVariable sigmoid(SDVariable iX) {
        return new Sigmoid(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable sigmoidDerivative(SDVariable iX, SDVariable wrt) {
        return new SigmoidDerivative(sameDiff(), iX, wrt).outputVariable();
    }


    public SDVariable logSigmoid(SDVariable iX) {
        return new LogSigmoid(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable logSigmoidDerivative(SDVariable iX, SDVariable wrt) {
        return new LogSigmoidDerivative(sameDiff(), iX, wrt).outputVariable();
    }

    public SDVariable powDerivative(SDVariable iX, double pow) {
        return new PowDerivative(sameDiff(), iX, false, pow).outputVariable();
    }


    public SDVariable swish(SDVariable iX) {
        return new Swish(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable swishDerivative(SDVariable iX) {
        return new SwishDerivative(sameDiff(), iX, false).outputVariable();
    }


    public SDVariable sign(SDVariable iX) {
        return new Sign(sameDiff(), iX, null).outputVariable();
    }


    public SDVariable expandDims(SDVariable iX, int axis) {
        return new ExpandDims(sameDiff(), new SDVariable[]{iX}, axis).outputVariable();
    }

    public SDVariable squeeze(SDVariable iX, int... axis) {
        return new Squeeze(sameDiff(), iX, axis).outputVariable();
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred) {
        return new ConfusionMatrix(sameDiff(), labels, pred).outputVariable();
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

    public SDVariable broadcast(SDVariable iX, int... shape) {
        return broadcast(iX, ArrayUtil.toLongArray(shape));
    }

    public SDVariable broadcast(SDVariable iX, long... shape) {
        return new Broadcast(sameDiff(), iX, shape).outputVariable();
    }

    public SDVariable onehot(SDVariable indices, int depth, int axis, double on, double off) {
        return new OneHot(sameDiff(), indices, depth, axis, on, off).outputVariable();
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
    
    public SDVariable assign(SDVariable x, Number num){
        return new ScalarSet(sameDiff(), x, num).outputVariable();
    }


    public SDVariable softsign(SDVariable iX) {
        return new SoftSign(sameDiff(), iX, null).outputVariable();

    }


    public SDVariable softsignDerivative(SDVariable iX) {
        return new SoftSignDerivative(sameDiff(), iX, null).outputVariable();

    }


    public SDVariable softplus(SDVariable iX) {
        return new SoftPlus(sameDiff(), iX, null).outputVariable();

    }


    public SDVariable elu(SDVariable iX) {
        return new ELU(sameDiff(), iX, null).outputVariable();

    }


    public SDVariable eluDerivative(SDVariable iX) {
        return new ELUDerivative(sameDiff(), iX, null).outputVariable();

    }


    public SDVariable leakyRelu(SDVariable iX, double cutoff) {
        return new LeakyReLU(sameDiff(), iX, false, cutoff).outputVariable();

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

    public SDVariable sequenceMask(SDVariable lengths, SDVariable maxLen) {
        return new SequenceMask(sameDiff(), lengths, maxLen).outputVariable();
    }

    public SDVariable sequenceMask(SDVariable lengths, int maxLen) {
        return new SequenceMask(sameDiff(), lengths, maxLen).outputVariable();
    }

    public SDVariable sequenceMask(SDVariable lengths) {
        return new SequenceMask(sameDiff(), lengths).outputVariable();
    }

    public SDVariable rollAxis(SDVariable iX, int axis) {
        return new RollAxis(sameDiff(), iX, axis).outputVariable();
    }

    public SDVariable concat(int dimension, SDVariable... inputs) {
        return new Concat(sameDiff(), dimension, inputs).outputVariable();
    }

    public SDVariable fill(SDVariable shape, double value) {
        return new Fill(sameDiff(), shape, value).outputVariable();
    }

    public SDVariable dot(SDVariable x, SDVariable y, int... dimensions){
        return new Dot(sameDiff(), x, y, dimensions).outputVariable();
    }

    public SDVariable[] dotBp(SDVariable in1, SDVariable in2, SDVariable grad, boolean keepDims, int... dimensions){
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

    public SDVariable sigmoidCrossEntropyWithLogits(SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        return new SigmoidCrossEntropyLoss(sameDiff(), logits, weights, labels,
                reductionMode, labelSmoothing).outputVariable();
    }

    public SDVariable softmaxCrossEntropyWithLogits(SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        return new SoftmaxCrossEntropyLoss(sameDiff(), logits, weights, labels,
                reductionMode, labelSmoothing).outputVariable();
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

    public SDVariable batchMmul(SDVariable[] matricesA,
                                SDVariable[] matricesB) {
        return batchMmul(matricesA, matricesB, false, false);
    }

    public SDVariable batchMmul(SDVariable[] matricesA,
                                SDVariable[] matricesB,
                                boolean transposeA,
                                boolean transposeB) {
        return batchMmul(ArrayUtils.addAll(matricesA, matricesB), transposeA, transposeB);
    }


    public SDVariable batchMmul(SDVariable[] matrices,
                                boolean transposeA,
                                boolean transposeB) {
        return new BatchMmul(sameDiff(), matrices, transposeA, transposeB).outputVariable();
    }


    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        validateDifferentialFunctionsameDiff(x);
        validateDifferentialFunctionsameDiff(y);
        return new TensorMmul(sameDiff(), x, y, dimensions).outputVariable();
    }


    public SDVariable softmaxDerivative(SDVariable functionInput, SDVariable wrt) {
        validateDifferentialFunctionsameDiff(functionInput);
        return new SoftMaxDerivative(sameDiff(), functionInput, wrt).outputVariable();
    }


    public SDVariable logSoftmax(SDVariable i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return new LogSoftMax(sameDiff(), i_v, null).outputVariable();

    }


    public SDVariable logSoftmaxDerivative(SDVariable arg, SDVariable wrt) {
        validateDifferentialFunctionsameDiff(arg);
        return new LogSoftMaxDerivative(sameDiff(), arg, wrt).outputVariable();
    }

    public SDVariable logSumExp(SDVariable arg, int... dimension){
        return new LogSumExp(sameDiff(), arg, dimension).outputVariable();
    }


    public SDVariable selu(SDVariable arg) {
        validateDifferentialFunctionsameDiff(arg);
        return new SELU(sameDiff(), arg, null).outputVariable();
    }


    public SDVariable seluDerivative(SDVariable arg) {
        validateDifferentialFunctionsameDiff(arg);
        return new SELUDerivative(sameDiff(), arg, null).outputVariable();
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

    public SDVariable setDiag(SDVariable in, SDVariable diag){
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

    public SDVariable dynamicStitch(SDVariable[] indices, SDVariable[] differentialFunctions) {
        for (SDVariable df : differentialFunctions)
            validateDifferentialFunctionsameDiff(df);

        return new DynamicStitch(sameDiff(), indices, differentialFunctions)
                .outputVariable();
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

    public SDVariable size(SDVariable in){
        return new Size(sameDiff(), in).outputVariable();
    }

    public SDVariable rank(SDVariable df){
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


    public SDVariable muli(SDVariable differentialFunction, SDVariable i_v) {
        validateDifferentialFunctionsameDiff(differentialFunction);
        return new MulOp(sameDiff(), new SDVariable[]{differentialFunction, i_v}, true).outputVariable();

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

    public SDVariable sliceBp(SDVariable input, SDVariable gradient, int[] begin, int[] size) {
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

    public SDVariable scatterUpdate(SDVariable ref, SDVariable indices, SDVariable updates) {
        return new ScatterUpdate(sameDiff(), ref, indices, updates).outputVariable();
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
