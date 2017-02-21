package org.deeplearning4j.nn.transferlearning;

import lombok.Builder;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;

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
        public Builder(){
        }

        public Builder seed(int seed){
            this.seed = (long)seed;
            return this;
        }

        public Builder seed(long seed){
            this.seed = seed;
            return this;
        }

        public Builder regularization(boolean regularization){
            this.useRegularization = regularization;
            return this;
        }

        public Builder iterations(int iterations){
            this.numIterations = iterations;
            return this;
        }

        public Builder activation(Activation activation){
            this.activationFn = activation.getActivationFunction();
            return this;
        }
    }

    public void applyToNeuralNetConfiguration(NeuralNetConfiguration nnc){

        Layer l = nnc.getLayer();
        if(l != null) {
            if (activationFn != null) l.setActivationFn(activationFn);
            if (weightInit != null) l.setWeightInit(weightInit);
            if (biasInit != null) l.setBiasInit(biasInit);
            if (dist != null) l.setDist(dist);
            if (learningRate != null) l.setLearningRate(learningRate);
            if (biasLearningRate != null) l.setBiasInit(biasLearningRate);
            if (learningRateSchedule != null) l.setLearningRateSchedule(learningRateSchedule);
//        if(lrScoreBasedDecay != null)
            if (l1 != null) l.setL1(l1);
            if (l2 != null) l.setL2(l2);
            if (l1Bias != null) l.setL1Bias(l1Bias);
            if (l2Bias != null) l.setL2Bias(l2Bias);
            if (dropOut != null) l.setDropOut(dropOut);
            if (updater != null) l.setUpdater(updater);
            if (momentum != null) l.setMomentum(momentum);
            if (momentumSchedule != null) l.setMomentum(momentum);
            if (epsilon != null) l.setEpsilon(epsilon);
            if (rho != null) l.setRho(rho);
            if (rmsDecay != null) l.setRmsDecay(rmsDecay);
            if (adamMeanDecay != null) l.setAdamMeanDecay(adamMeanDecay);
            if (adamVarDecay != null) l.setAdamVarDecay(adamVarDecay);
        }
        if(miniBatch != null) nnc.setMiniBatch(miniBatch);
        if(numIterations != null) nnc.setNumIterations(numIterations);
        if(maxNumLineSearchIterations != null) nnc.setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if(seed != null) nnc.setSeed(seed);
        if(useRegularization != null) nnc.setUseRegularization(useRegularization);
        if(optimizationAlgo != null) nnc.setOptimizationAlgo(optimizationAlgo);
        if(stepFunction != null) nnc.setStepFunction(stepFunction);
        if(useDropConnect != null) nnc.setUseDropConnect(useDropConnect);
        if(minimize != null) nnc.setMinimize(minimize);
        if(gradientNormalization != null) l.setGradientNormalization(gradientNormalization);
        if(gradientNormalizationThreshold != null) l.setGradientNormalizationThreshold(gradientNormalizationThreshold);
        if(learningRatePolicy != null) nnc.setLearningRatePolicy(learningRatePolicy);
        if(lrPolicySteps != null) nnc.setLrPolicySteps(lrPolicySteps);
        if(lrPolicyPower != null) nnc.setLrPolicyPower(lrPolicyPower);

        if(l instanceof ConvolutionLayer){
            ((ConvolutionLayer)l).setConvolutionMode(convolutionMode);
        }
        if(l instanceof SubsamplingLayer){
            ((SubsamplingLayer)l).setConvolutionMode(convolutionMode);
        }
    }

    public void applyToMultiLayerConfiguration(MultiLayerConfiguration conf){

        if(pretrain != null) conf.setPretrain(pretrain);
        if(backprop != null) conf.setBackprop(backprop);
        if(backpropType != null) conf.setBackpropType(backpropType);
        if(tbpttFwdLength != null) conf.setTbpttFwdLength(tbpttFwdLength);
        if(tbpttBackLength != null) conf.setTbpttBackLength(tbpttBackLength);
    }

    public void applyToComputationGraphConfiguration(ComputationGraphConfiguration conf){

        throw new RuntimeException("TODO");
    }


    public NeuralNetConfiguration toNeuralNetConfiguration(){
        //This gives us a NNC with the defaults set - and any appropriate overrides set
        NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder().build();
        applyToNeuralNetConfiguration(nnc);
        return nnc;
    }
}
