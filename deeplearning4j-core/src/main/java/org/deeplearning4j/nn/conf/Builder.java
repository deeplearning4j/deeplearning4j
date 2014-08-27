package org.deeplearning4j.nn.conf;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.api.NeuralNetwork;

import java.util.Map;

public class Builder {

    private float sparsity = 1e-1f;
    private boolean useAdaGrad = true;
    private float lr = 1e-1f;
    private float momentum = 0;
    private float l2 = 2e-4f;
    private boolean useRegularization = true;
    private Map<Integer, Float> momentumAfter;
    private int resetAdaGradIterations = -1;
    private float dropOut = 0;
    private boolean applySparsity = false;
    private WeightInit weightInit = WeightInit.SI;
    private NeuralNetwork.OptimizationAlgorithm optimizationAlgo = NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT;
    private int renderWeightsEveryNumEpochs = -1;
    private boolean concatBiases = false;
    private boolean constrainGradientToUnitNorm = false;
    private RandomGenerator rng = new MersenneTwister(123);
    private long seed = 123;
    private RealDistribution dist  = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    private boolean adagrad = true;
    private LossFunctions.LossFunction lossFunction;
    private int nIn;
    private int nOut;
    private ActivationFunction activationFunction = Activations.sigmoid();
    private boolean useHiddenActivationsForwardProp = true;
    private RBM.VisibleUnit visibleUnit = RBM.VisibleUnit.BINARY;
    private RBM.HiddenUnit hiddenUnit = RBM.HiddenUnit.BINARY;

    public Builder dist(RealDistribution dist) {
        this.dist = dist;
        return this;
    }


    public Builder sparsity(float sparsity) {
        this.sparsity = sparsity;
        return this;
    }

    public Builder useAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
        return this;
    }

    public Builder learningRate(float lr) {
        this.lr = lr;
        return this;
    }

    public Builder momentum(float momentum) {
        this.momentum = momentum;
        return this;
    }

    public Builder regularizationCoefficient(float l2) {
        this.l2 = l2;
        return this;
    }

    public Builder regularize(boolean useRegularization) {
        this.useRegularization = useRegularization;
        return this;
    }

    public Builder momentumAfter(Map<Integer, Float> momentumAfter) {
        this.momentumAfter = momentumAfter;
        return this;
    }

    public Builder adagradResetIterations(int resetAdaGradIterations) {
        this.resetAdaGradIterations = resetAdaGradIterations;
        return this;
    }

    public Builder dropOut(float dropOut) {
        this.dropOut = dropOut;
        return this;
    }

    public Builder applySparsity(boolean applySparsity) {
        this.applySparsity = applySparsity;
        return this;
    }

    public Builder weightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
        return this;
    }

    public Builder optimize(NeuralNetwork.OptimizationAlgorithm optimizationAlgo) {
        this.optimizationAlgo = optimizationAlgo;
        return this;
    }

    public Builder render(int renderWeightsEveryNumEpochs) {
        this.renderWeightsEveryNumEpochs = renderWeightsEveryNumEpochs;
        return this;
    }

    public Builder concatBiases(boolean concatBiases) {
        this.concatBiases = concatBiases;
        return this;
    }

    public Builder constrainWeights(boolean constrainGradientToUnitNorm) {
        this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
        return this;
    }

    public Builder rng(RandomGenerator rng) {
        this.rng = rng;
        return this;
    }

    public Builder seed(long seed) {
        this.seed = seed;
        return this;
    }

    public NeuralNetConfiguration build() {
        return new NeuralNetConfiguration(sparsity,useAdaGrad,lr,momentum,l2,useRegularization,momentumAfter,resetAdaGradIterations,dropOut,applySparsity,weightInit,optimizationAlgo,lossFunction,renderWeightsEveryNumEpochs,concatBiases,constrainGradientToUnitNorm,rng,dist,seed,nIn,nOut,activationFunction,useHiddenActivationsForwardProp,visibleUnit,hiddenUnit);

     }

    public Builder setSparsity(float sparsity) {
        this.sparsity = sparsity;
        return this;
    }

    public Builder setUseAdaGrad(boolean adagrad) {
        this.adagrad = adagrad;
        return this;
    }

    public Builder setLr(float lr) {
        this.lr = lr;
        return this;
    }

    public Builder setMomentum(float momentum) {
        this.momentum = momentum;
        return this;
    }

    public Builder setL2(float l2) {
        this.l2 = l2;
        return this;
    }

    public Builder setUseRegularization(boolean useRegularization) {
        this.useRegularization = useRegularization;
        return this;
    }

    public Builder setMomentumAfter(Map<Integer, Float> momentumAfter) {
        this.momentumAfter = momentumAfter;
        return this;
    }

    public Builder setResetAdaGradIterations(int resetAdaGradIterations) {
        this.resetAdaGradIterations = resetAdaGradIterations;
        return this;
    }

    public Builder setDropOut(float dropOut) {
        this.dropOut = dropOut;
        return this;
    }

    public Builder setApplySparsity(boolean applySparsity) {
        this.applySparsity = applySparsity;
        return this;
    }

    public Builder setWeightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
        return this;
    }

    public Builder setOptimizationAlgo(NeuralNetwork.OptimizationAlgorithm optimizationAlgo) {
        this.optimizationAlgo = optimizationAlgo;
        return this;
    }

    public Builder setLossFunction(LossFunctions.LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        return this;
    }

    public Builder setRenderWeightsEveryNumEpochs(int renderWeightsEveryNumEpochs) {
        this.renderWeightsEveryNumEpochs = renderWeightsEveryNumEpochs;
        return this;
    }

    public Builder setConcatBiases(boolean concatBiases) {
        this.concatBiases = concatBiases;
        return this;
    }

    public Builder setConstrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
        this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
        return this;
    }

    public Builder setRng(RandomGenerator rng) {
        this.rng = rng;
        return this;
    }

    public Builder setDist(RealDistribution dist) {
        this.dist = dist;
        return this;
    }

    public Builder setSeed(long seed) {
        this.seed = seed;
        return this;
    }

    public Builder setnIn(int nIn) {
        this.nIn = nIn;
        return this;
    }

    public Builder setnOut(int nOut) {
        this.nOut = nOut;
        return this;
    }

    public Builder setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    public Builder setUseHiddenActivationsForwardProp(boolean useHiddenActivationsForwardProp) {
        this.useHiddenActivationsForwardProp = useHiddenActivationsForwardProp;
        return this;
    }

    public Builder setVisibleUnit(RBM.VisibleUnit visibleUnit) {
        this.visibleUnit = visibleUnit;
        return this;
    }

    public Builder setHiddenUnit(RBM.HiddenUnit hiddenUnit) {
        this.hiddenUnit = hiddenUnit;
        return this;
    }
}