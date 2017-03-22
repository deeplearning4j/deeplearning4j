package org.deeplearning4j.nn.transferlearning;

import lombok.Builder;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;

import java.util.Map;

/**
 * Created by Alex on 21/02/2017.
 */
@Builder(builderClassName = "Builder")
public class FineTuneConfiguration {

    protected IActivation activationFn;
    protected WeightInit weightInit;
    protected Double biasInit;
    protected Distribution dist;
    protected Double learningRate;
    protected Double biasLearningRate;
    protected Map<Integer, Double> learningRateSchedule;
    protected Double lrScoreBasedDecay;
    protected Double l1;
    protected Double l2;
    protected Double l1Bias;
    protected Double l2Bias;
    protected Double dropOut;
    protected Updater updater;
    protected Double momentum;
    protected Map<Integer, Double> momentumSchedule;
    protected Double epsilon;
    protected Double rho;
    protected Double rmsDecay;
    protected Double adamMeanDecay;
    protected Double adamVarDecay;
    protected Boolean miniBatch;
    protected Integer numIterations;
    protected Integer maxNumLineSearchIterations;
    protected Long seed;
    protected Boolean useRegularization;
    protected OptimizationAlgorithm optimizationAlgo;
    protected StepFunction stepFunction;
    protected Boolean useDropConnect;
    protected Boolean minimize;
    protected GradientNormalization gradientNormalization;
    protected Double gradientNormalizationThreshold;
    protected LearningRatePolicy learningRatePolicy;
    protected Double lrPolicyDecayRate;
    protected Double lrPolicySteps;
    protected Double lrPolicyPower;
    protected ConvolutionMode convolutionMode;

    protected Boolean pretrain;
    protected Boolean backprop;
    protected BackpropType backpropType;
    protected Integer tbpttFwdLength;
    protected Integer tbpttBackLength;

    //Lombok builder. Note that the code below ADDS OR OVERRIDES the lombok implementation; the final builder class
    // is the composite of the lombok parts and the parts defined here
    //partial implementation to allow public no-arg constructor (lombok default is package private)
    //Plus some implementations to match NeuralNetConfiguration builder methods
    public static class Builder {
        public Builder() {}

        public Builder seed(int seed) {
            this.seed = (long) seed;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder regularization(boolean regularization) {
            this.useRegularization = regularization;
            return this;
        }

        public Builder iterations(int iterations) {
            this.numIterations = iterations;
            return this;
        }

        public Builder activation(Activation activation) {
            this.activationFn = activation.getActivationFunction();
            return this;
        }
    }


    public NeuralNetConfiguration appliedNeuralNetConfiguration(NeuralNetConfiguration nnc) {
        applyToNeuralNetConfiguration(nnc);
        nnc = new NeuralNetConfiguration.Builder(nnc.clone()).build();
        return nnc;
    }

    public void applyToNeuralNetConfiguration(NeuralNetConfiguration nnc) {

        Layer l = nnc.getLayer();
        Updater originalUpdater = null;
        WeightInit origWeightInit = null;
        if (l != null) {
            originalUpdater = l.getUpdater();
            origWeightInit = l.getWeightInit();
            if (activationFn != null)
                l.setActivationFn(activationFn);
            if (weightInit != null)
                l.setWeightInit(weightInit);
            if (biasInit != null)
                l.setBiasInit(biasInit);
            if (dist != null)
                l.setDist(dist);
            if (learningRate != null) {
                //usually the same learning rate is applied to both bias and weights
                //so always overwrite the learning rate to both?
                l.setLearningRate(learningRate);
                l.setBiasLearningRate(learningRate);
            }
            if (biasLearningRate != null)
                l.setBiasLearningRate(biasLearningRate);
            if (learningRateSchedule != null)
                l.setLearningRateSchedule(learningRateSchedule);
            //        if(lrScoreBasedDecay != null)
            if (l1 != null)
                l.setL1(l1);
            if (l2 != null)
                l.setL2(l2);
            if (l1Bias != null)
                l.setL1Bias(l1Bias);
            if (l2Bias != null)
                l.setL2Bias(l2Bias);
            if (dropOut != null)
                l.setDropOut(dropOut);
            if (updater != null)
                l.setUpdater(updater);
            if (momentum != null)
                l.setMomentum(momentum);
            if (momentumSchedule != null)
                l.setMomentum(momentum);
            if (epsilon != null)
                l.setEpsilon(epsilon);
            if (rho != null)
                l.setRho(rho);
            if (rmsDecay != null)
                l.setRmsDecay(rmsDecay);
            if (adamMeanDecay != null)
                l.setAdamMeanDecay(adamMeanDecay);
            if (adamVarDecay != null)
                l.setAdamVarDecay(adamVarDecay);
        }
        if (miniBatch != null)
            nnc.setMiniBatch(miniBatch);
        if (numIterations != null)
            nnc.setNumIterations(numIterations);
        if (maxNumLineSearchIterations != null)
            nnc.setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if (seed != null)
            nnc.setSeed(seed);
        if (useRegularization != null)
            nnc.setUseRegularization(useRegularization);
        if (optimizationAlgo != null)
            nnc.setOptimizationAlgo(optimizationAlgo);
        if (stepFunction != null)
            nnc.setStepFunction(stepFunction);
        if (useDropConnect != null)
            nnc.setUseDropConnect(useDropConnect);
        if (minimize != null)
            nnc.setMinimize(minimize);
        if (gradientNormalization != null)
            l.setGradientNormalization(gradientNormalization);
        if (gradientNormalizationThreshold != null)
            l.setGradientNormalizationThreshold(gradientNormalizationThreshold);
        if (learningRatePolicy != null)
            nnc.setLearningRatePolicy(learningRatePolicy);
        if (lrPolicySteps != null)
            nnc.setLrPolicySteps(lrPolicySteps);
        if (lrPolicyPower != null)
            nnc.setLrPolicyPower(lrPolicyPower);

        if (convolutionMode != null && l instanceof ConvolutionLayer) {
            ((ConvolutionLayer) l).setConvolutionMode(convolutionMode);
        }
        if (convolutionMode != null && l instanceof SubsamplingLayer) {
            ((SubsamplingLayer) l).setConvolutionMode(convolutionMode);
        }

        //Check the updater config. If we change updaters, we want to remove the old config to avoid warnings
        if (l != null && updater != null && originalUpdater != null && updater != originalUpdater) {
            switch (originalUpdater) {
                case ADAM:
                    if (adamMeanDecay == null)
                        l.setAdamMeanDecay(Double.NaN);
                    if (adamVarDecay == null)
                        l.setAdamVarDecay(Double.NaN);
                    break;
                case ADADELTA:
                    if (rho == null)
                        l.setRho(Double.NaN);
                    if (epsilon == null)
                        l.setEpsilon(Double.NaN);
                    break;
                case NESTEROVS:
                    if (momentum == null)
                        l.setMomentum(Double.NaN);
                    if (momentumSchedule == null)
                        l.setMomentumSchedule(null);
                    if (epsilon == null)
                        l.setEpsilon(Double.NaN);
                    break;
                case ADAGRAD:
                    if (epsilon == null)
                        l.setEpsilon(Double.NaN);
                    break;
                case RMSPROP:
                    if (rmsDecay == null)
                        l.setRmsDecay(Double.NaN);
                    if (epsilon == null)
                        l.setEpsilon(Double.NaN);
                    break;

                //Other cases: no changes required
            }
        }

        //Check weight init. Remove dist if originally was DISTRIBUTION, and isn't now -> remove no longer needed distribution
        if (l != null && origWeightInit == WeightInit.DISTRIBUTION && weightInit != null
                        && weightInit != WeightInit.DISTRIBUTION) {
            l.setDist(null);
        }

        //Perform validation. This also sets the defaults for updaters. For example, Updater.RMSProp -> set rmsDecay
        if (l != null) {
            LayerValidation.updaterValidation(l.getLayerName(), l, momentum, momentumSchedule, adamMeanDecay,
                            adamVarDecay, rho, rmsDecay, epsilon);

            boolean useDropCon = (useDropConnect == null ? nnc.isUseDropConnect() : useDropConnect);
            LayerValidation.generalValidation(l.getLayerName(), l, nnc.isUseRegularization(), useDropCon, dropOut, l2,
                            l2Bias, l1, l1Bias, dist);
        }

        //Also: update the LR, L1 and L2 maps, based on current config (which might be different to original config)
        if (nnc.variables(false) != null) {
            for (String s : nnc.variables(false)) {
                nnc.setLayerParamLR(s);
            }
        }
    }

    public void applyToMultiLayerConfiguration(MultiLayerConfiguration conf) {
        if (pretrain != null)
            conf.setPretrain(pretrain);
        if (backprop != null)
            conf.setBackprop(backprop);
        if (backpropType != null)
            conf.setBackpropType(backpropType);
        if (tbpttFwdLength != null)
            conf.setTbpttFwdLength(tbpttFwdLength);
        if (tbpttBackLength != null)
            conf.setTbpttBackLength(tbpttBackLength);
    }

    public void applyToComputationGraphConfiguration(ComputationGraphConfiguration conf) {
        if (pretrain != null)
            conf.setPretrain(pretrain);
        if (backprop != null)
            conf.setBackprop(backprop);
        if (backpropType != null)
            conf.setBackpropType(backpropType);
        if (tbpttFwdLength != null)
            conf.setTbpttFwdLength(tbpttFwdLength);
        if (tbpttBackLength != null)
            conf.setTbpttBackLength(tbpttBackLength);
    }

    public NeuralNetConfiguration.Builder appliedNeuralNetConfigurationBuilder() {
        NeuralNetConfiguration.Builder confBuilder = new NeuralNetConfiguration.Builder();
        if (activationFn != null)
            confBuilder.setActivationFn(activationFn);
        if (weightInit != null)
            confBuilder.setWeightInit(weightInit);
        if (biasInit != null)
            confBuilder.setBiasInit(biasInit);
        if (dist != null)
            confBuilder.setDist(dist);
        if (learningRate != null) {
            //usually the same learning rate is applied to both bias and weights
            //so always overwrite the learning rate to both?
            confBuilder.setLearningRate(learningRate);
            confBuilder.setBiasLearningRate(learningRate);
        }
        if (biasLearningRate != null)
            confBuilder.setBiasLearningRate(biasLearningRate);
        if (learningRateSchedule != null)
            confBuilder.setLearningRateSchedule(learningRateSchedule);
        //      if(lrScoreBasedDecay != null)
        if (l1 != null)
            confBuilder.setL1(l1);
        if (l2 != null)
            confBuilder.setL2(l2);
        if (l1Bias != null)
            confBuilder.setL1Bias(l1Bias);
        if (l2Bias != null)
            confBuilder.setL2Bias(l2Bias);
        if (dropOut != null)
            confBuilder.setDropOut(dropOut);
        if (updater != null)
            confBuilder.setUpdater(updater);
        if (momentum != null)
            confBuilder.setMomentum(momentum);
        if (momentumSchedule != null)
            confBuilder.setMomentum(momentum);
        if (epsilon != null)
            confBuilder.setEpsilon(epsilon);
        if (rho != null)
            confBuilder.setRho(rho);
        if (rmsDecay != null)
            confBuilder.setRmsDecay(rmsDecay);
        if (adamMeanDecay != null)
            confBuilder.setAdamMeanDecay(adamMeanDecay);
        if (adamVarDecay != null)
            confBuilder.setAdamVarDecay(adamVarDecay);
        if (miniBatch != null)
            confBuilder.setMiniBatch(miniBatch);
        if (numIterations != null)
            confBuilder.setNumIterations(numIterations);
        if (maxNumLineSearchIterations != null)
            confBuilder.setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if (seed != null)
            confBuilder.setSeed(seed);
        if (useRegularization != null)
            confBuilder.setUseRegularization(useRegularization);
        if (optimizationAlgo != null)
            confBuilder.setOptimizationAlgo(optimizationAlgo);
        if (stepFunction != null)
            confBuilder.setStepFunction(stepFunction);
        if (useDropConnect != null)
            confBuilder.setUseDropConnect(useDropConnect);
        if (minimize != null)
            confBuilder.setMinimize(minimize);
        if (gradientNormalization != null)
            confBuilder.setGradientNormalization(gradientNormalization);
        if (gradientNormalizationThreshold != null)
            confBuilder.setGradientNormalizationThreshold(gradientNormalizationThreshold);
        if (learningRatePolicy != null)
            confBuilder.setLearningRatePolicy(learningRatePolicy);
        if (lrPolicySteps != null)
            confBuilder.setLrPolicySteps(lrPolicySteps);
        if (lrPolicyPower != null)
            confBuilder.setLrPolicyPower(lrPolicyPower);

        return confBuilder;
    }
}
