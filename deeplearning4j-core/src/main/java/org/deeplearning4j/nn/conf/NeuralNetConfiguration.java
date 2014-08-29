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

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * A Serializable configuration
 * for neural nets that covers per layer parameters
 *
 * @author Adam Gibson
 */
public class NeuralNetConfiguration implements Serializable,Cloneable {

    private float sparsity = 0f;
    private boolean useAdaGrad = true;
    private float lr = 1e-1f;
    protected int k = 1;
    protected float corruptionLevel = 0.3f;
    protected int numIterations = 1000;
    /* momentum for learning */
    protected float momentum = 0.5f;
    /* L2 Regularization constant */
    protected float l2 = 0f;
    protected boolean useRegularization = false;
    //momentum after n iterations
    protected Map<Integer,Float> momentumAfter = new HashMap<>();
    //reset adagrad historical gradient after n iterations
    protected int resetAdaGradIterations = -1;
    protected float dropOut = 0;
    //use only when binary hidden neuralNets are active
    protected boolean applySparsity = false;
    //weight init scheme, this can either be a distribution or a applyTransformToDestination scheme
    protected WeightInit weightInit;
    protected NeuralNetwork.OptimizationAlgorithm optimizationAlgo = NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT;
    protected LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY;
    protected int renderWeightsEveryNumEpochs = -1;
    //whether to concat hidden bias or add it
    protected  boolean concatBiases = false;
    //whether to constrain the gradient to unit norm or not
    protected boolean constrainGradientToUnitNorm = false;
    /* RNG for sampling. */
    protected transient RandomGenerator rng;
    protected transient RealDistribution dist;
    protected long seed = 123;
    protected int nIn,nOut;
    protected ActivationFunction activationFunction;
    protected boolean useHiddenActivationsForwardProp = true;
    private RBM.VisibleUnit visibleUnit = RBM.VisibleUnit.BINARY;
    private RBM.HiddenUnit hiddenUnit = RBM.HiddenUnit.BINARY;

    public NeuralNetConfiguration() {

    }



    public NeuralNetConfiguration(float sparsity, boolean useAdaGrad, float lr, float momentum, float l2, boolean useRegularization, Map<Integer, Float> momentumAfter, int resetAdaGradIterations, float dropOut, boolean applySparsity, WeightInit weightInit, NeuralNetwork.OptimizationAlgorithm optimizationAlgo, LossFunctions.LossFunction lossFunction, int renderWeightsEveryNumEpochs, boolean concatBiases, boolean constrainGradientToUnitNorm, RandomGenerator rng, RealDistribution dist, long seed, int nIn, int nOut, ActivationFunction activationFunction, boolean useHiddenActivationsForwardProp, RBM.VisibleUnit visibleUnit, RBM.HiddenUnit hiddenUnit) {
        this.sparsity = sparsity;
        this.useAdaGrad = useAdaGrad;
        this.lr = lr;
        this.momentum = momentum;
        this.l2 = l2;
        this.useRegularization = useRegularization;
        this.momentumAfter = momentumAfter;
        this.resetAdaGradIterations = resetAdaGradIterations;
        this.dropOut = dropOut;
        this.applySparsity = applySparsity;
        this.weightInit = weightInit;
        this.optimizationAlgo = optimizationAlgo;
        this.lossFunction = lossFunction;
        this.renderWeightsEveryNumEpochs = renderWeightsEveryNumEpochs;
        this.concatBiases = concatBiases;
        this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
        this.rng = rng;
        this.dist = dist;
        this.seed = seed;
        this.nIn = nIn;
        this.nOut = nOut;
        this.activationFunction = activationFunction;
        this.useHiddenActivationsForwardProp = useHiddenActivationsForwardProp;
        this.visibleUnit = visibleUnit;
        if(dist == null)
            this.dist = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

        this.hiddenUnit = hiddenUnit;
    }

    public NeuralNetConfiguration(NeuralNetConfiguration neuralNetConfiguration) {
        this.sparsity = neuralNetConfiguration.sparsity;
        this.useAdaGrad = neuralNetConfiguration.useAdaGrad;
        this.lr = neuralNetConfiguration.lr;
        this.momentum = neuralNetConfiguration.momentum;
        this.l2 = l2;
        this.useRegularization = neuralNetConfiguration.useRegularization;
        this.momentumAfter = neuralNetConfiguration.momentumAfter;
        this.resetAdaGradIterations = neuralNetConfiguration.resetAdaGradIterations;
        this.dropOut = neuralNetConfiguration.dropOut;
        this.applySparsity = neuralNetConfiguration.applySparsity;
        this.weightInit = neuralNetConfiguration.weightInit;
        this.optimizationAlgo = neuralNetConfiguration.optimizationAlgo;
        this.lossFunction = neuralNetConfiguration.lossFunction;
        this.renderWeightsEveryNumEpochs = neuralNetConfiguration.renderWeightsEveryNumEpochs;
        this.concatBiases = neuralNetConfiguration.concatBiases;
        this.constrainGradientToUnitNorm = neuralNetConfiguration.constrainGradientToUnitNorm;
        this.rng = neuralNetConfiguration.rng;
        this.dist = neuralNetConfiguration.dist;
        this.seed = neuralNetConfiguration.seed;
        this.nIn = neuralNetConfiguration.nIn;
        this.nOut = neuralNetConfiguration.nOut;
        this.activationFunction = neuralNetConfiguration.activationFunction;
        this.useHiddenActivationsForwardProp = neuralNetConfiguration.useHiddenActivationsForwardProp;
        this.visibleUnit = neuralNetConfiguration.visibleUnit;
        if(dist == null)
            this.dist = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

        this.hiddenUnit = neuralNetConfiguration.hiddenUnit;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public void setNumIterations(int numIterations) {
        this.numIterations = numIterations;
    }

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    public float getCorruptionLevel() {
        return corruptionLevel;
    }

    public void setCorruptionLevel(float corruptionLevel) {
        this.corruptionLevel = corruptionLevel;
    }

    public RBM.HiddenUnit getHiddenUnit() {
        return hiddenUnit;
    }

    public void setHiddenUnit(RBM.HiddenUnit hiddenUnit) {
        this.hiddenUnit = hiddenUnit;
    }

    public RBM.VisibleUnit getVisibleUnit() {
        return visibleUnit;
    }

    public void setVisibleUnit(RBM.VisibleUnit visibleUnit) {
        this.visibleUnit = visibleUnit;
    }

    public boolean isUseHiddenActivationsForwardProp() {
        return useHiddenActivationsForwardProp;
    }

    public void setUseHiddenActivationsForwardProp(boolean useHiddenActivationsForwardProp) {
        this.useHiddenActivationsForwardProp = useHiddenActivationsForwardProp;
    }

    public LossFunctions.LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunctions.LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public int getnIn() {
        return nIn;
    }

    public void setnIn(int nIn) {
        this.nIn = nIn;
    }

    public int getnOut() {
        return nOut;
    }

    public void setnOut(int nOut) {
        this.nOut = nOut;
    }

    public float getSparsity() {
        return sparsity;
    }

    public void setSparsity(float sparsity) {
        this.sparsity = sparsity;
    }

    public boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public void setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }

    public float getLr() {
        return lr;
    }

    public void setLr(float lr) {
        this.lr = lr;
    }

    public float getMomentum() {
        return momentum;
    }

    public void setMomentum(float momentum) {
        this.momentum = momentum;
    }

    public float getL2() {
        return l2;
    }

    public void setL2(float l2) {
        this.l2 = l2;
    }

    public boolean isUseRegularization() {
        return useRegularization;
    }

    public void setUseRegularization(boolean useRegularization) {
        this.useRegularization = useRegularization;
    }

    public Map<Integer, Float> getMomentumAfter() {
        return momentumAfter;
    }

    public void setMomentumAfter(Map<Integer, Float> momentumAfter) {
        this.momentumAfter = momentumAfter;
    }

    public int getResetAdaGradIterations() {
        return resetAdaGradIterations;
    }

    public void setResetAdaGradIterations(int resetAdaGradIterations) {
        this.resetAdaGradIterations = resetAdaGradIterations;
    }

    public float getDropOut() {
        return dropOut;
    }

    public void setDropOut(float dropOut) {
        this.dropOut = dropOut;
    }

    public boolean isApplySparsity() {
        return applySparsity;
    }

    public void setApplySparsity(boolean applySparsity) {
        this.applySparsity = applySparsity;
    }

    public WeightInit getWeightInit() {
        return weightInit;
    }

    public void setWeightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
    }

    public NeuralNetwork.OptimizationAlgorithm getOptimizationAlgo() {
        return optimizationAlgo;
    }

    public void setOptimizationAlgo(NeuralNetwork.OptimizationAlgorithm optimizationAlgo) {
        this.optimizationAlgo = optimizationAlgo;
    }

    public int getRenderWeightsEveryNumEpochs() {
        return renderWeightsEveryNumEpochs;
    }

    public void setRenderWeightsEveryNumEpochs(int renderWeightsEveryNumEpochs) {
        this.renderWeightsEveryNumEpochs = renderWeightsEveryNumEpochs;
    }

    public boolean isConcatBiases() {
        return concatBiases;
    }

    public void setConcatBiases(boolean concatBiases) {
        this.concatBiases = concatBiases;
    }

    public boolean isConstrainGradientToUnitNorm() {
        return constrainGradientToUnitNorm;
    }

    public void setConstrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
        this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
    }

    public RandomGenerator getRng() {
        return rng;
    }

    public void setRng(RandomGenerator rng) {
        this.rng = rng;
    }

    public long getSeed() {
        return seed;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }

    public RealDistribution getDist() {
        return dist;
    }

    public void setDist(RealDistribution dist) {
        this.dist = dist;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NeuralNetConfiguration)) return false;

        NeuralNetConfiguration that = (NeuralNetConfiguration) o;

        if (applySparsity != that.applySparsity) return false;
        if (concatBiases != that.concatBiases) return false;
        if (constrainGradientToUnitNorm != that.constrainGradientToUnitNorm) return false;
        if (Float.compare(that.dropOut, dropOut) != 0) return false;
        if (Float.compare(that.l2, l2) != 0) return false;
        if (Float.compare(that.lr, lr) != 0) return false;
        if (Float.compare(that.momentum, momentum) != 0) return false;
        if (nIn != that.nIn) return false;
        if (nOut != that.nOut) return false;
        if (renderWeightsEveryNumEpochs != that.renderWeightsEveryNumEpochs) return false;
        if (resetAdaGradIterations != that.resetAdaGradIterations) return false;
        if (seed != that.seed) return false;
        if (Float.compare(that.sparsity, sparsity) != 0) return false;
        if (useAdaGrad != that.useAdaGrad) return false;
        if (useRegularization != that.useRegularization) return false;
        if (!activationFunction.equals(that.activationFunction)) return false;
        if (dist != null ? !dist.equals(that.dist) : that.dist != null) return false;
        if (momentumAfter != null ? !momentumAfter.equals(that.momentumAfter) : that.momentumAfter != null)
            return false;
        if (optimizationAlgo != that.optimizationAlgo) return false;
        if (rng != null ? !rng.equals(that.rng) : that.rng != null) return false;
        if (weightInit != that.weightInit) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (sparsity != +0.0f ? Float.floatToIntBits(sparsity) : 0);
        result = 31 * result + (useAdaGrad ? 1 : 0);
        result = 31 * result + (lr != +0.0f ? Float.floatToIntBits(lr) : 0);
        result = 31 * result + (momentum != +0.0f ? Float.floatToIntBits(momentum) : 0);
        result = 31 * result + (l2 != +0.0f ? Float.floatToIntBits(l2) : 0);
        result = 31 * result + (useRegularization ? 1 : 0);
        result = 31 * result + (momentumAfter != null ? momentumAfter.hashCode() : 0);
        result = 31 * result + resetAdaGradIterations;
        result = 31 * result + (dropOut != +0.0f ? Float.floatToIntBits(dropOut) : 0);
        result = 31 * result + (applySparsity ? 1 : 0);
        result = 31 * result + weightInit.hashCode();
        result = 31 * result + optimizationAlgo.hashCode();
        result = 31 * result + renderWeightsEveryNumEpochs;
        result = 31 * result + (concatBiases ? 1 : 0);
        result = 31 * result + (constrainGradientToUnitNorm ? 1 : 0);
        result = 31 * result + (rng != null ? rng.hashCode() : 0);
        result = 31 * result + (dist != null ? dist.hashCode() : 0);
        result = 31 * result + (int) (seed ^ (seed >>> 32));
        result = 31 * result + nIn;
        result = 31 * result + nOut;
        result = 31 * result + activationFunction.hashCode();
        return result;
    }

    /**
     * Creates and returns a copy of this object.  The precise meaning
     * of "copy" may depend on the class of the object. The general
     * intent is that, for any object {@code x}, the expression:
     * <blockquote>
     * <pre>
     * x.clone() != x</pre></blockquote>
     * will be true, and that the expression:
     * <blockquote>
     * <pre>
     * x.clone().getClass() == x.getClass()</pre></blockquote>
     * will be {@code true}, but these are not absolute requirements.
     * While it is typically the case that:
     * <blockquote>
     * <pre>
     * x.clone().equals(x)</pre></blockquote>
     * will be {@code true}, this is not an absolute requirement.
     *
     * By convention, the returned object should be obtained by calling
     * {@code super.clone}.  If a class and all of its superclasses (except
     * {@code Object}) obey this convention, it will be the case that
     * {@code x.clone().getClass() == x.getClass()}.
     *
     * By convention, the object returned by this method should be independent
     * of this object (which is being cloned).  To achieve this independence,
     * it may be necessary to modify one or more fields of the object returned
     * by {@code super.clone} before returning it.  Typically, this means
     * copying any mutable objects that comprise the internal "deep structure"
     * of the object being cloned and replacing the references to these
     * objects with references to the copies.  If a class contains only
     * primitive fields or references to immutable objects, then it is usually
     * the case that no fields in the object returned by {@code super.clone}
     * need to be modified.
     *
     * The method {@code clone} for class {@code Object} performs a
     * specific cloning operation. First, if the class of this object does
     * not implement the interface {@code Cloneable}, then a
     * {@code CloneNotSupportedException} is thrown. Note that all arrays
     * are considered to implement the interface {@code Cloneable} and that
     * the return type of the {@code clone} method of an array type {@code T[]}
     * is {@code T[]} where T is any reference or primitive type.
     * Otherwise, this method creates a new instance of the class of this
     * object and initializes all its fields with exactly the contents of
     * the corresponding fields of this object, as if by assignment; the
     * contents of the fields are not themselves cloned. Thus, this method
     * performs a "shallow copy" of this object, not a "deep copy" operation.
     *
     * The class {@code Object} does not itself implement the interface
     * {@code Cloneable}, so calling the {@code clone} method on an object
     * whose class is {@code Object} will result in throwing an
     * exception at run time.
     *
     * @return a clone of this instance.
     * @throws CloneNotSupportedException if the object's class does not
     *                                    support the {@code Cloneable} interface. Subclasses
     *                                    that override the {@code clone} method can also
     *                                    throw this exception to indicate that an instance cannot
     *                                    be cloned.
     * @see Cloneable
     */
    @Override
    public NeuralNetConfiguration clone()  {
        return new NeuralNetConfiguration(this);
    }

    public static class Builder {

        private float sparsity = 0f;
        private boolean useAdaGrad = true;
        private float lr = 1e-1f;
        private float momentum = 0;
        private float l2 = 0f;
        private boolean useRegularization = false;
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
        private LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY;
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






        public Builder l2(float l2) {
            this.l2 = l2;
            return this;
        }

        public Builder regularization(boolean useRegularization) {
            this.useRegularization = useRegularization;
            return this;
        }



        public Builder resetAdaGradIterations(int resetAdaGradIterations) {
            this.resetAdaGradIterations = resetAdaGradIterations;
            return this;
        }







        public Builder optimizationAlgo(NeuralNetwork.OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        public Builder lossFunction(LossFunctions.LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        public Builder renderWeightsEveryNumEpochs(int renderWeightsEveryNumEpochs) {
            this.renderWeightsEveryNumEpochs = renderWeightsEveryNumEpochs;
            return this;
        }



        public Builder constrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
            this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
            return this;
        }





        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }

        public Builder activationFunction(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder useHiddenActivationsForwardProp(boolean useHiddenActivationsForwardProp) {
            this.useHiddenActivationsForwardProp = useHiddenActivationsForwardProp;
            return this;
        }

        public Builder visibleUnit(RBM.VisibleUnit visibleUnit) {
            this.visibleUnit = visibleUnit;
            return this;
        }

        public Builder hiddenUnit(RBM.HiddenUnit hiddenUnit) {
            this.hiddenUnit = hiddenUnit;
            return this;
        }
    }
}
