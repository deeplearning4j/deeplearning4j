package org.deeplearning4j.nn.transferlearning;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.primitives.Optional;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;
import java.util.List;

/**
 * Configuration for fine tuning. Note that values set here will override values for all non-frozen layers
 *
 * @author Alex Black
 * @author Susan Eraly
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonInclude(JsonInclude.Include.NON_NULL)
@NoArgsConstructor
@AllArgsConstructor
@Data
public class FineTuneConfiguration {

    protected IActivation activationFn;
    protected WeightInit weightInit;
    protected Double biasInit;
    protected Distribution dist;
    protected Double l1;
    protected Double l2;
    protected Double l1Bias;
    protected Double l2Bias;
    protected Optional<IDropout> dropout;
    protected Optional<IWeightNoise> weightNoise;
    protected IUpdater updater;
    protected IUpdater biasUpdater;
    protected Boolean miniBatch;
    protected Integer maxNumLineSearchIterations;
    protected Long seed;
    protected OptimizationAlgorithm optimizationAlgo;
    protected StepFunction stepFunction;
    protected Boolean minimize;
    protected Optional<GradientNormalization> gradientNormalization;
    protected Double gradientNormalizationThreshold;
    protected ConvolutionMode convolutionMode;
    protected Optional<List<LayerConstraint>> constraints;

    protected Boolean pretrain;
    protected Boolean backprop;
    protected BackpropType backpropType;
    protected Integer tbpttFwdLength;
    protected Integer tbpttBackLength;

    protected WorkspaceMode trainingWorkspaceMode;
    protected WorkspaceMode inferenceWorkspaceMode;

    public static Builder builder() {
        return new Builder();
    }

    /*
     * Can't use Lombok @Builder annotation due to optionals (otherwise we have a bunch of ugly .x(Optional<T> value)
      * methods - lombok builder doesn't support excluding fields? :(
     * Note the use of optional here: gives us 3 states...
     * 1. Null: not set
     * 2. Optional (empty): set to null
     * 3. Optional (not empty): set to specific value
     *
     * Obviously, having null only makes sense for some things (dropout, etc) whereas null for other things doesn't
     * make sense
     */
    public static class Builder {
        private IActivation activation;
        private WeightInit weightInit;
        private Double biasInit;
        private Distribution dist;
        private Double l1;
        private Double l2;
        private Double l1Bias;
        private Double l2Bias;
        private Optional<IDropout> dropout;
        private Optional<IWeightNoise> weightNoise;
        private IUpdater updater;
        private IUpdater biasUpdater;
        private Boolean miniBatch;
        private Integer maxNumLineSearchIterations;
        private Long seed;
        private OptimizationAlgorithm optimizationAlgo;
        private StepFunction stepFunction;
        private Boolean minimize;
        private Optional<GradientNormalization> gradientNormalization;
        private Double gradientNormalizationThreshold;
        private ConvolutionMode convolutionMode;
        private Optional<List<LayerConstraint>> constraints;
        private Boolean pretrain;
        private Boolean backprop;
        private BackpropType backpropType;
        private Integer tbpttFwdLength;
        private Integer tbpttBackLength;
        private WorkspaceMode trainingWorkspaceMode;
        private WorkspaceMode inferenceWorkspaceMode;

        public Builder() {

        }

        public Builder activation(IActivation activationFn) {
            this.activation = activationFn;
            return this;
        }

        public Builder activation(Activation activation) {
            this.activation = activation.getActivationFunction();
            return this;
        }

        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        public Builder biasInit(double biasInit) {
            this.biasInit = biasInit;
            return this;
        }

        public Builder dist(Distribution dist) {
            this.dist = dist;
            return this;
        }

        public Builder l1(double l1) {
            this.l1 = l1;
            return this;
        }

        public Builder l2(double l2) {
            this.l2 = l2;
            return this;
        }

        public Builder l1Bias(double l1Bias) {
            this.l1Bias = l1Bias;
            return this;
        }

        public Builder l2Bias(double l2Bias) {
            this.l2Bias = l2Bias;
            return this;
        }

        public Builder dropout(IDropout dropout) {
            this.dropout = Optional.ofNullable(dropout);
            return this;
        }

        public Builder dropOut(double dropout){
            if(dropout == 0.0){
                return dropout(null);
            }
            return dropout(new Dropout(dropout));
        }

        public Builder weightNoise(IWeightNoise weightNoise) {
            this.weightNoise = Optional.ofNullable(weightNoise);
            return this;
        }

        public Builder updater(IUpdater updater) {
            this.updater = updater;
            return this;
        }

        @Deprecated
        public Builder updater(Updater updater) {
            return updater(updater.getIUpdaterWithDefaultConfig());
        }

        public Builder biasUpdater(IUpdater biasUpdater) {
            this.biasUpdater = biasUpdater;
            return this;
        }

        public Builder miniBatch(boolean miniBatch) {
            this.miniBatch = miniBatch;
            return this;
        }

        public Builder maxNumLineSearchIterations(int maxNumLineSearchIterations) {
            this.maxNumLineSearchIterations = maxNumLineSearchIterations;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder seed(int seed){
            return seed((long)seed);
        }

        public Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        public Builder stepFunction(StepFunction stepFunction) {
            this.stepFunction = stepFunction;
            return this;
        }

        public Builder minimize(boolean minimize) {
            this.minimize = minimize;
            return this;
        }

        public Builder gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = Optional.ofNullable(gradientNormalization);
            return this;
        }

        public Builder gradientNormalizationThreshold(double gradientNormalizationThreshold) {
            this.gradientNormalizationThreshold = gradientNormalizationThreshold;
            return this;
        }

        public Builder convolutionMode(ConvolutionMode convolutionMode) {
            this.convolutionMode = convolutionMode;
            return this;
        }

        public Builder constraints(List<LayerConstraint> constraints) {
            this.constraints = Optional.ofNullable(constraints);
            return this;
        }

        public Builder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        public Builder backprop(boolean backprop) {
            this.backprop = backprop;
            return this;
        }

        public Builder backpropType(BackpropType backpropType) {
            this.backpropType = backpropType;
            return this;
        }

        public Builder tbpttFwdLength(int tbpttFwdLength) {
            this.tbpttFwdLength = tbpttFwdLength;
            return this;
        }

        public Builder tbpttBackLength(int tbpttBackLength) {
            this.tbpttBackLength = tbpttBackLength;
            return this;
        }

        public Builder trainingWorkspaceMode(WorkspaceMode trainingWorkspaceMode) {
            this.trainingWorkspaceMode = trainingWorkspaceMode;
            return this;
        }

        public Builder inferenceWorkspaceMode(WorkspaceMode inferenceWorkspaceMode) {
            this.inferenceWorkspaceMode = inferenceWorkspaceMode;
            return this;
        }

        public FineTuneConfiguration build() {
            return new FineTuneConfiguration(activation, weightInit, biasInit, dist, l1, l2, l1Bias, l2Bias, dropout, weightNoise, updater, biasUpdater, miniBatch, maxNumLineSearchIterations, seed, optimizationAlgo, stepFunction, minimize, gradientNormalization, gradientNormalizationThreshold, convolutionMode, constraints, pretrain, backprop, backpropType, tbpttFwdLength, tbpttBackLength, trainingWorkspaceMode, inferenceWorkspaceMode);
        }

        public String toString() {
            return "FineTuneConfiguration.Builder(activation=" + this.activation + ", weightInit=" + this.weightInit + ", biasInit=" + this.biasInit + ", dist=" + this.dist + ", l1=" + this.l1 + ", l2=" + this.l2 + ", l1Bias=" + this.l1Bias + ", l2Bias=" + this.l2Bias + ", dropout=" + this.dropout + ", weightNoise=" + this.weightNoise + ", updater=" + this.updater + ", biasUpdater=" + this.biasUpdater + ", miniBatch=" + this.miniBatch + ", maxNumLineSearchIterations=" + this.maxNumLineSearchIterations + ", seed=" + this.seed + ", optimizationAlgo=" + this.optimizationAlgo + ", stepFunction=" + this.stepFunction + ", minimize=" + this.minimize + ", gradientNormalization=" + this.gradientNormalization + ", gradientNormalizationThreshold=" + this.gradientNormalizationThreshold + ", convolutionMode=" + this.convolutionMode + ", constraints=" + this.constraints + ", pretrain=" + this.pretrain + ", backprop=" + this.backprop + ", backpropType=" + this.backpropType + ", tbpttFwdLength=" + this.tbpttFwdLength + ", tbpttBackLength=" + this.tbpttBackLength + ", trainingWorkspaceMode=" + this.trainingWorkspaceMode + ", inferenceWorkspaceMode=" + this.inferenceWorkspaceMode + ")";
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
            if (dropout != null)
                l.setIDropout(dropout.orElse(null));
            if(constraints != null)
                l.setConstraints(constraints.orElse(null));
        }

        if (l != null && l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            origWeightInit = bl.getWeightInit();
            if (activationFn != null)
                bl.setActivationFn(activationFn);
            if (weightInit != null)
                bl.setWeightInit(weightInit);
            if (biasInit != null)
                bl.setBiasInit(biasInit);
            if (dist != null)
                bl.setDist(dist);
            if (l1 != null)
                bl.setL1(l1);
            if (l2 != null)
                bl.setL2(l2);
            if (l1Bias != null)
                bl.setL1Bias(l1Bias);
            if (l2Bias != null)
                bl.setL2Bias(l2Bias);
            if (gradientNormalization != null)
                bl.setGradientNormalization(gradientNormalization.orElse(null));
            if (gradientNormalizationThreshold != null)
                bl.setGradientNormalizationThreshold(gradientNormalizationThreshold);
            if (updater != null){
                bl.setIUpdater(updater);
            }
            if (biasUpdater != null){
                bl.setBiasUpdater(biasUpdater);
            }
            if (weightNoise != null){
                bl.setWeightNoise(weightNoise.orElse(null));
            }
        }
        if (miniBatch != null)
            nnc.setMiniBatch(miniBatch);
        if (maxNumLineSearchIterations != null)
            nnc.setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if (seed != null)
            nnc.setSeed(seed);
        if (optimizationAlgo != null)
            nnc.setOptimizationAlgo(optimizationAlgo);
        if (stepFunction != null)
            nnc.setStepFunction(stepFunction);
        if (minimize != null)
            nnc.setMinimize(minimize);

        if (convolutionMode != null && l instanceof ConvolutionLayer) {
            ((ConvolutionLayer) l).setConvolutionMode(convolutionMode);
        }
        if (convolutionMode != null && l instanceof SubsamplingLayer) {
            ((SubsamplingLayer) l).setConvolutionMode(convolutionMode);
        }

        //Check weight init. Remove dist if originally was DISTRIBUTION, and isn't now -> remove no longer needed distribution
        if (l != null && l instanceof BaseLayer && origWeightInit == WeightInit.DISTRIBUTION && weightInit != null
                        && weightInit != WeightInit.DISTRIBUTION) {
            ((BaseLayer) l).setDist(null);
        }

        //Perform validation
        if (l != null) {
            LayerValidation.generalValidation(l.getLayerName(), l, get(dropout), l2, l2Bias, l1, l1Bias,
                    dist, get(constraints), null, null);
        }

        //Also: update the LR, L1 and L2 maps, based on current config (which might be different to original config)
        if (nnc.variables(false) != null) {
            for (String s : nnc.variables(false)) {
                nnc.setLayerParamLR(s);
            }
        }
    }

    private static <T> T get(Optional<T> optional){
        if(optional == null){
            return null;
        }
        return optional.orElse(null);
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
        if (l1 != null)
            confBuilder.setL1(l1);
        if (l2 != null)
            confBuilder.setL2(l2);
        if (l1Bias != null)
            confBuilder.setL1Bias(l1Bias);
        if (l2Bias != null)
            confBuilder.setL2Bias(l2Bias);
        if (dropout != null)
            confBuilder.setIdropOut(dropout.orElse(null));
        if (updater != null)
            confBuilder.updater(updater);
        if(biasUpdater != null)
            confBuilder.biasUpdater(biasUpdater);
        if (miniBatch != null)
            confBuilder.setMiniBatch(miniBatch);
        if (maxNumLineSearchIterations != null)
            confBuilder.setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if (seed != null)
            confBuilder.setSeed(seed);
        if (optimizationAlgo != null)
            confBuilder.setOptimizationAlgo(optimizationAlgo);
        if (stepFunction != null)
            confBuilder.setStepFunction(stepFunction);
        if (minimize != null)
            confBuilder.setMinimize(minimize);
        if (gradientNormalization != null)
            confBuilder.setGradientNormalization(gradientNormalization.orElse(null));
        if (gradientNormalizationThreshold != null)
            confBuilder.setGradientNormalizationThreshold(gradientNormalizationThreshold);
        if (trainingWorkspaceMode != null)
            confBuilder.trainingWorkspaceMode(trainingWorkspaceMode);
        if (inferenceWorkspaceMode != null)
            confBuilder.inferenceWorkspaceMode(inferenceWorkspaceMode);
        return confBuilder;
    }


    public String toJson() {
        try {
            return NeuralNetConfiguration.mapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public String toYaml() {
        try {
            return NeuralNetConfiguration.mapperYaml().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public static FineTuneConfiguration fromJson(String json) {
        try {
            return NeuralNetConfiguration.mapper().readValue(json, FineTuneConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static FineTuneConfiguration fromYaml(String yaml) {
        try {
            return NeuralNetConfiguration.mapperYaml().readValue(yaml, FineTuneConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
