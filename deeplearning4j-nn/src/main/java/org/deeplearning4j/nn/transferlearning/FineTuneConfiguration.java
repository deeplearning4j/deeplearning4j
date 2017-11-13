package org.deeplearning4j.nn.transferlearning;

import lombok.AllArgsConstructor;
import lombok.Builder;
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
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;
import java.util.List;

/**
 * Created by Alex on 21/02/2017.
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonInclude(JsonInclude.Include.NON_NULL)
@NoArgsConstructor
@AllArgsConstructor
@Data
@Builder(builderClassName = "Builder")
public class FineTuneConfiguration {

    protected IActivation activationFn;
    protected WeightInit weightInit;
    protected Double biasInit;
    protected Distribution dist;
    protected Double l1;
    protected Double l2;
    protected Double l1Bias;
    protected Double l2Bias;
    protected IDropout dropout;
    protected IWeightNoise weightNoise;
    protected IUpdater iUpdater;
    protected IUpdater biasUpdater;
    protected Boolean miniBatch;
    protected Integer numIterations;
    protected Integer maxNumLineSearchIterations;
    protected Long seed;
    protected OptimizationAlgorithm optimizationAlgo;
    protected StepFunction stepFunction;
    protected Boolean minimize;
    protected GradientNormalization gradientNormalization;
    protected Double gradientNormalizationThreshold;
    protected ConvolutionMode convolutionMode;
    protected List<LayerConstraint> constraints;
    protected Boolean hasBiasConstraints;
    protected Boolean hasWeightConstraints;

    protected Boolean pretrain;
    protected Boolean backprop;
    protected BackpropType backpropType;
    protected Integer tbpttFwdLength;
    protected Integer tbpttBackLength;

    protected WorkspaceMode trainingWorkspaceMode;
    protected WorkspaceMode inferenceWorkspaceMode;

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

        public Builder iterations(int iterations) {
            this.numIterations = iterations;
            return this;
        }

        public Builder dropOut(double dropout){
            return dropout(new Dropout(dropout));
        }

        public Builder activation(Activation activation) {
            this.activationFn = activation.getActivationFunction();
            return this;
        }

        public Builder updater(IUpdater updater) {
            return iUpdater(updater);
        }

        @Deprecated
        public Builder updater(Updater updater) {
            return updater(updater.getIUpdaterWithDefaultConfig());
        }
    }


    void applyToLayer(Layer l) {

        WeightInit origWeightInit = null;

        if (l != null) {
            if (dropout != null)
                l.setIDropout(dropout);
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
                bl.setGradientNormalization(gradientNormalization);
            if (gradientNormalizationThreshold != null)
                bl.setGradientNormalizationThreshold(gradientNormalizationThreshold);
            if (iUpdater != null){
                bl.setIUpdater(iUpdater);
            }
            if (biasUpdater != null){
                bl.setBiasUpdater(biasUpdater);
            }
            if (weightNoise != null){
                bl.setWeightNoise(weightNoise);
            }
        }


        // Check weight init. Remove dist if originally was DISTRIBUTION, and isn't now
        // -> remove no longer needed distribution
        if (l != null && l instanceof BaseLayer && origWeightInit == WeightInit.DISTRIBUTION && weightInit != null
                && weightInit != WeightInit.DISTRIBUTION) {
            ((BaseLayer) l).setDist(null);
        }

        // Perform validation. This also sets the defaults for updaters.
        // For example, Updater.RMSProp -> set rmsDecay
        if (l != null) {
            LayerValidation.generalValidation(l.getLayerName(), l, dropout, l2, l2Bias,
                    l1, l1Bias, dist, constraints, null, null);
        }

        if (convolutionMode != null && l instanceof ConvolutionLayer) {
            ((ConvolutionLayer) l).setConvolutionMode(convolutionMode);
        }
        if (convolutionMode != null && l instanceof SubsamplingLayer) {
            ((SubsamplingLayer) l).setConvolutionMode(convolutionMode);
        }

    }


    public void applyToMultiLayerConfiguration(MultiLayerConfiguration config) {
        if (miniBatch != null)
            config.setMiniBatch(miniBatch);
        if (maxNumLineSearchIterations != null)
            config.setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if (seed != null)
            config.setSeed(seed);
        if (optimizationAlgo != null)
            config.setOptimizationAlgo(optimizationAlgo);
        if (stepFunction != null)
            config.setStepFunction(stepFunction);
        if (minimize != null)
            config.setMinimize(minimize);
        if (pretrain != null)
            config.setPretrain(pretrain);
        if (backprop != null)
            config.setBackprop(backprop);
        if (backpropType != null)
            config.setBackpropType(backpropType);
        if (tbpttFwdLength != null)
            config.setTbpttFwdLength(tbpttFwdLength);
        if (tbpttBackLength != null)
            config.setTbpttBackLength(tbpttBackLength);
    }

    public void applyToComputationGraphConfiguration(ComputationGraphConfiguration config) {
        if (miniBatch != null)
            config.setMiniBatch(miniBatch);
        if (maxNumLineSearchIterations != null)
            config.setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if (seed != null)
            config.setSeed(seed);
        if (optimizationAlgo != null)
            config.setOptimizationAlgo(optimizationAlgo);
        if (stepFunction != null)
            config.setStepFunction(stepFunction);
        if (minimize != null)
            config.setMinimize(minimize);
        if (pretrain != null)
            config.setPretrain(pretrain);
        if (backprop != null)
            config.setBackprop(backprop);
        if (backpropType != null)
            config.setBackpropType(backpropType);
        if (tbpttFwdLength != null)
            config.setTbpttFwdLength(tbpttFwdLength);
        if (tbpttBackLength != null)
            config.setTbpttBackLength(tbpttBackLength);
    }

    NeuralNetConfiguration.Builder appliedNeuralNetConfigurationBuilder() {
        NeuralNetConfiguration.Builder confBuilder = new NeuralNetConfiguration.Builder();
        if (activationFn != null)
            confBuilder.getGlobalConf().setActivationFn(activationFn);
        if (weightInit != null)
            confBuilder.getGlobalConf().setWeightInit(weightInit);
        if (biasInit != null)
            confBuilder.getGlobalConf().setBiasInit(biasInit);
        if (dist != null)
            confBuilder.getGlobalConf().setDist(dist);
        if (l1 != null)
            confBuilder.getGlobalConf().setL1(l1);
        if (l2 != null)
            confBuilder.getGlobalConf().setL2(l2);
        if (l1Bias != null)
            confBuilder.getGlobalConf().setL1Bias(l1Bias);
        if (l2Bias != null)
            confBuilder.getGlobalConf().setL2Bias(l2Bias);
        if (dropout != null)
            confBuilder.getGlobalConf().setDropOut(dropout);
        if (iUpdater != null)
            confBuilder.getGlobalConf().setUpdater(iUpdater);
        if(biasUpdater != null)
            confBuilder.getGlobalConf().setBiasUpdater(biasUpdater);
        if (miniBatch != null)
            confBuilder.getGlobalConf().setMiniBatch(miniBatch);
        if (maxNumLineSearchIterations != null)
            confBuilder.getGlobalConf().setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if (seed != null)
            confBuilder.getGlobalConf().setSeed(seed);
        if (optimizationAlgo != null)
            confBuilder.getGlobalConf().setOptimizationAlgo(optimizationAlgo);
        if (stepFunction != null)
            confBuilder.getGlobalConf().setStepFunction(stepFunction);
        if (minimize != null)
            confBuilder.getGlobalConf().setMinimize(minimize);
        if (gradientNormalization != null)
            confBuilder.getGlobalConf().setGradientNormalization(gradientNormalization);
        if (gradientNormalizationThreshold != null)
            confBuilder.getGlobalConf().setGradientNormalizationThreshold(gradientNormalizationThreshold);
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
