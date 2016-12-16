package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by susaneraly on 12/14/16.
 * @author susaneraly
 */
public class ActivationParamsHelper {

    /**
     * Default order for the arrays created by activation function parameter initialization
     */
    public static final char DEFAULT_ACTIVATION_PARAM_INIT_ORDER = 'f';
    private static final Logger log = LoggerFactory.getLogger(ActivationParamsHelper.class);
    private final static String ACTIVATION_PARAM_KEY_PREFIX = "aP_";

    /**
     * Does the specified activation function have learnable parameters?
     * @param configuration NeuralNet conf
     * @return has learnable params or not
     */
    public static boolean hasParams(NeuralNetConfiguration configuration) {
        return numParams(configuration) != 0;
    }

    /**
     * The number of learnable parameters in the activation function
     * @param configuration NeuralNet conf
     * @return the number of learnable params in the activation function
     */
    public static int numParams(NeuralNetConfiguration configuration) {
        return configuration.getLayer().getActivationFn().getNumParams();
    }

    public static int expandedNumParams(NeuralNetConfiguration configuration, int [] activationShape) {

        if (!hasParams(configuration)) {
            return 0;
        }

        checkSettings(configuration);

        boolean [] sharedParams = configuration.getLayer().getActivationFn().isSharedParam();
        boolean [] shardedParams = configuration.getLayer().getActivationFn().isShardedParam();
        int [] shardAcrossDim = configuration.getLayer().getActivationFn().getShardAcrossDim();

        int expandedParamCount = 0;
        expandedParamCount += trueCount(sharedParams);
        expandedParamCount += trueCount(shardedParams) * totalElements(activationShape);
        expandedParamCount += dimensionShardLength(sharedParams,shardedParams,shardAcrossDim,activationShape);

        return expandedParamCount;

    }

    public static int lengthParamI(int i, NeuralNetConfiguration configuration, int [] activationShape) {

        boolean [] sharedParams = configuration.getLayer().getActivationFn().isSharedParam();
        boolean [] shardedParams = configuration.getLayer().getActivationFn().isShardedParam();
        int [] shardAcrossDim = configuration.getLayer().getActivationFn().getShardAcrossDim();

        if (sharedParams[i]) {
            return 1;
        }
        else if (shardedParams[i]) {
            return totalElements(activationShape);
        }
        else {
            int defaultDim = 1;
            if (shardAcrossDim != null) {
                int index = indexConvert(i,sharedParams,shardedParams);
                return dimensionShardLength(shardAcrossDim[index],activationShape);
            }
            else {
                return dimensionShardLength(defaultDim,activationShape);
            }
        }
    }

    public static int[] shapeParamI(int i, NeuralNetConfiguration configuration, int [] activationShape) {
        if (configuration.getLayer().getActivationFn().isSharedParam()[i]) {
            return new int[]{1, 1};
        }
        else {
            return activationShape;
        }
    }

    /**
     * Reshape the parameters view, without modifying the paramsView array values.
     * Same reshaping mechanism as {@link #initWeights(int[], WeightInit, Distribution, INDArray)}
     *
     * @param shape      Shape to reshape
     * @param paramsView Parameters array view
     */
    public static INDArray reshapeWeights(int[] shape, INDArray paramsView) {
        return reshapeWeights(shape, paramsView, DEFAULT_WEIGHT_INIT_ORDER);
    }

    /**
     * Reshape the parameters view, without modifying the paramsView array values.
     * Same reshaping mechanism as {@link #initWeights(int[], WeightInit, Distribution, char, INDArray)}
     *
     * @param shape           Shape to reshape
     * @param paramsView      Parameters array view
     * @param flatteningOrder Order in which parameters are flattened/reshaped
     */
    public static INDArray reshapeWeights(int[] shape, INDArray paramsView, char flatteningOrder) {
        return paramsView.reshape(flatteningOrder, shape);
    }

    protected static INDArray initParam(NeuralNetConfiguration configuration, int i, int [] activationShape) {
        return configuration.getLayer().getActivationFn().initParam(i,shapeParamI(i,configuration,activationShape));
    }

    private static int checkSettings(NeuralNetConfiguration configuration) {
        return checkSettings(configuration.getLayer().getActivationFn());
    }

    public static int checkSettings(IActivation afn) {

        int totalParams = afn.getNumParams();
        boolean [] sharedParams = afn.isSharedParam();
        boolean [] shardedParams = afn.isShardedParam();
        int [] shardAcrossDim = afn.getShardAcrossDim();

        if (sharedParams.length != totalParams || shardedParams.length != totalParams) {
            throw new IllegalStateException("Activation Function implementation error! \"" + afn.getClass().getName()+ "\" length of the sharedParams and shardedParams array should be equal to the number of params specified ");
        }

        log.info("There are a total of " + totalParams +" learnable params in the defined activation function");

        for (int i=0; i<totalParams;i++) {
            if (sharedParams[i] && sharedParams[i]) {
                throw new IllegalStateException("Learnable parameter at index "+i+": Activation params cannot be specified as both shared and sharded!\n" +
                                                "Shared params are shared across all activation nodes.\n" +
                                                "A sharded parameter will be replicated for each activation node.");
            }
            if (sharedParams[i]) {
                log.info("Learnable parameter at index "+i+" is a shared param. It will shared across all activation nodes");

            }
            else if (shardedParams[i]) {
                log.info("Learnable parameter at index "+i+" is sharded and will be replicated at each activation node.");
            }
            else {
                if(shardAcrossDim != null) {
                    if (bothFalseCount(sharedParams,sharedParams) != shardAcrossDim.length)  {
                        throw new IllegalStateException("If specifying a dimension to shard along, it must be specified for all parameters not shared across all dimensions or sharded across all dimensions.");
                    }

                    log.info("Learnable parameter at index " + i + " is specified as sharded across dimension "+shardAcrossDim[i]);
                }
                else {
                    log.info("Learnable parameter at index " + i + " is specified as not shared by all or sharded across all. Will default to sharding across features for time series and channels for images.");
                }
            }
        }

    }

    private static int totalElements(int [] activationShape) {
        int total = 1;
        for (int i=0; i<activationShape.length;i++) {
            total *= activationShape[i];
        }
        return total;
    }

    private static int dimensionShardLength(boolean [] settingA, boolean [] settingB, int[] dimensionAlong, int [] activationShape) {
        int total = 0;
        int defaultDim  = 1; //channels for images, features for time series
        if (dimensionAlong != null) {
            for (int i = 0; i < bothFalseCount(settingA, settingB); i++) {
                //dimensions are checked for correctness before here
                total += dimensionShardLength(dimensionAlong[i], activationShape);
            }
        }
        else {
            total += bothFalseCount(settingA,settingB) * dimensionShardLength(defaultDim,activationShape);
        }
        return total;
    }

    private static int dimensionShardLength(int dimension, int [] activationShape) {
        int total = 1;
        for (int i=0; i<activationShape.length; i++) {
            if (i!=dimension) total *= activationShape[i];
        }
        return total;
    }

    private static int trueCount(boolean setting[]) {
        int total = 0;
        for (int i=0; i<setting.length; i++) {
            if(setting[i]) i++;
        }
        return total;
    }

    private static int bothFalseCount(boolean [] settingA, boolean [] settingB) {
        int total = 0;
        for (int i=0; i<settingA.length;i++) {
            if(!settingA[i] && !settingB[i]) total++;
        }
        return total;
    }

    private static int indexConvert(int index, boolean [] settingA, boolean [] settingB) {
        int newIndex = 0;
        for (int i=0; i<index;i++) {
            if(!settingA[i] && !settingB[i]) newIndex++;
        }
        return newIndex;
    }
}
