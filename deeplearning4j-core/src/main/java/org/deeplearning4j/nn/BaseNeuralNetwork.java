package org.deeplearning4j.nn;


import static org.deeplearning4j.linalg.ops.transforms.Transforms.*;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.lossfunctions.*;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.optimize.NeuralNetworkOptimizer;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.util.Dl4jReflection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Baseline class for any Neural Network used
 * as a layer in a deep network such as an {@link DBN}
 * @author Adam Gibson
 *
 */
public abstract class BaseNeuralNetwork implements NeuralNetwork,Persistable {




    private static final long serialVersionUID = -7074102204433996574L;
    /* Number of visible inputs */
    protected int nVisible;
    /**
     * Number of hidden units
     * One tip with this is usually having
     * more hidden units than inputs (read: input rows here)
     * will typically cause terrible overfitting.
     *
     * Another rule worthy of note: more training data typically results
     * in more redundant data. It is usually a better idea to use a smaller number
     * of hidden units.
     *
     *
     *
     **/
    protected int nHidden;
    /* Weight matrix */
    protected INDArray W;
    /* hidden bias */
    protected INDArray hBias;
    /* visible bias */
    protected INDArray vBias;
    /* RNG for sampling. */
    protected RandomGenerator rng;
    /* input to the network */
    protected INDArray input;
    /* sparsity target */
    protected double sparsity = 0;
    /* momentum for learning */
    protected double momentum = 0.5;
    /* Probability distribution for weights */
    protected transient RealDistribution dist = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    /* L2 Regularization constant */
    protected double l2 = 0.1;
    protected transient NeuralNetworkOptimizer optimizer;
    protected int renderWeightsEveryNumEpochs = -1;
    protected double fanIn = -1;
    protected boolean useRegularization = false;
    protected boolean useAdaGrad = true;
    //used to track if adagrad needs to be changed
    protected boolean firstTimeThrough = false;
    //normalize by input rows or not
    protected boolean normalizeByInputRows = true;
    //use only when binary hidden layers are active
    protected boolean applySparsity = false;
    protected double dropOut = 0;
    protected INDArray doMask;
    protected INDArray gvMask;
    protected OptimizationAlgorithm optimizationAlgo;
    protected LossFunction lossFunction;
    private static Logger log = LoggerFactory.getLogger(BaseNeuralNetwork.class);
    //cache input when training?
    protected boolean cacheInput;
    //previous gradient used for updates
    protected INDArray wGradient,vBiasGradient,hBiasGradient;

    protected int lastMiniBatchSize = 1;

    //momentum after n iterations
    protected Map<Integer,Double> momentumAfter = new HashMap<>();
    //reset adagrad historical gradient after n iterations
    protected int resetAdaGradIterations = -1;

    //adaptive learning rate for each of the biases and weights
    protected AdaGrad wAdaGrad,hBiasAdaGrad,vBiasAdaGrad;
    //whether to concat hidden bias or add it
    protected  boolean concatBiases = false;
    //whether to constrain the gradient to unit norm or not
    protected boolean constrainGradientToUnitNorm = false;
    //weight init scheme, this can either be a distribution or a applyTransformToDestination scheme
    protected WeightInit weightInit;


    protected BaseNeuralNetwork() {}
    /**
     *
     * @param nVisible the number of outbound nodes
     * @param nHidden the number of nodes in the hidden layer
     * @param W the weights for this vector, maybe null, if so this will
     * createComplex a matrix with nHidden x nVisible dimensions.
     * @param rng the rng, if not a seed of 1234 is used.
     */
    public BaseNeuralNetwork(int nVisible, int nHidden,
                             INDArray W, INDArray hbias, INDArray vbias, RandomGenerator rng,double fanIn,RealDistribution dist) {
        this(null,nVisible,nHidden,W,hbias,vbias,rng,fanIn,dist);

    }

    /**
     *
     * @param input the input examples
     * @param nVisible the number of outbound nodes
     * @param nHidden the number of nodes in the hidden layer
     * @param W the weights for this vector, maybe null, if so this will
     * createComplex a matrix with nHidden x nVisible dimensions.
     * @param rng the rng, if not a seed of 1234 is used.
     */
    public BaseNeuralNetwork(INDArray input, int nVisible, int nHidden,
                             INDArray W, INDArray hbias, INDArray vbias, RandomGenerator rng,double fanIn,RealDistribution dist) {
        this.nVisible = nVisible;
        if(dist != null)
            this.dist = dist;
        else
            this.dist = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
        this.nHidden = nHidden;
        this.fanIn = fanIn;
        this.input = input;
        if(rng == null)
            this.rng = new MersenneTwister(1234);

        else
            this.rng = rng;
        this.W = W;
        if(this.W != null)
            this.wAdaGrad = new AdaGrad(this.W.rows(),this.W.columns());

        this.vBias = vbias;
        if(this.vBias != null)
            this.vBiasAdaGrad = new AdaGrad(this.vBias.rows(),this.vBias.columns());


        this.hBias = hbias;
        if(this.hBias != null)
            this.hBiasAdaGrad = new AdaGrad(this.hBias.rows(),this.hBias.columns());


        initWeights();


    }

    /**
     * Whether to cache the input at training time
     *
     * @return true if the input should be cached at training time otherwise false
     */
    @Override
    public boolean cacheInput() {
        return cacheInput;
    }

    @Override
    public double l2RegularizedCoefficient() {
        return ((double) pow(getW(),2).sum(Integer.MAX_VALUE).element()/ 2.0)  * l2 + 1e-6;
    }

    /**
     * Initialize weights.
     * This includes steps for doing a random initialization of W
     * as well as the vbias and hbias
     */
    protected void initWeights()  {

        if(this.nVisible < 1)
            throw new IllegalStateException("Number of visible can not be less than 1");
        if(this.nHidden < 1)
            throw new IllegalStateException("Number of hidden can not be less than 1");

        if(this.dist == null)
            dist = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
		/*
		 * Initialize based on the number of visible units..
		 * The lower bound is called the fan in
		 * The outer bound is called the fan out.
		 * 
		 * Below's advice works for Denoising AutoEncoders and other 
		 * neural networks you will use due to the same baseline guiding principles for
		 * both RBMs and Denoising Autoencoders.
		 * 
		 * Hinton's Guide to practical RBMs:
		 * The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
		 * a standard deviation of about 0.01. Using larger random values can speed the initial learning, but
		 * it may lead to a slightly worse final model. Care should be taken to ensure that the initial weight
		 * values do not allow typical visible vectors to drive the hidden unit probabilities very close to 1 or 0
		 * as this significantly slows the learning.
		 */
        if(this.W == null) {

            this.W = NDArrays.zeros(nVisible,nHidden);

            for(int i = 0; i < this.W.rows(); i++)
                this.W.putRow(i,NDArrays.create(dist.sample(this.W.columns())));

        }

        this.wAdaGrad = new AdaGrad(this.W.rows(),this.W.columns());

        if(this.hBias == null) {
            this.hBias = NDArrays.zeros(nHidden);
			/*
			 * Encourage sparsity.
			 * See Hinton's Practical guide to RBMs
			 */
            //this.hBias.subi(4);
        }

        this.hBiasAdaGrad = new AdaGrad(hBias.rows(),hBias.columns());


        if(this.vBias == null) {
            if(this.input != null) {

                this.vBias = NDArrays.zeros(nVisible);


            }
            else
                this.vBias = NDArrays.zeros(nVisible);
        }

        this.vBiasAdaGrad = new AdaGrad(vBias.rows(),vBias.columns());


    }




    @Override
    public void resetAdaGrad(double lr) {
        if(!firstTimeThrough) {
            this.wAdaGrad = new AdaGrad(this.getW().rows(),this.getW().columns(),lr);
            firstTimeThrough = false;
        }

    }

    public void setRenderEpochs(int renderEpochs) {
        this.renderWeightsEveryNumEpochs = renderEpochs;

    }
    @Override
    public int getRenderIterations() {
        return renderWeightsEveryNumEpochs;
    }

    @Override
    public double fanIn() {
        return fanIn < 0 ? 1 / nVisible : fanIn;
    }

    @Override
    public void setFanIn(double fanIn) {
        this.fanIn = fanIn;
    }



    /**
     * Backprop with the output being the reconstruction
     */
    @Override
    public void backProp(double lr,int iterations,Object[] extraParams) {
        double currRecon = squaredLoss();
        boolean train = true;
        NeuralNetwork revert = clone();
        while(train) {
            if(iterations > iterations)
                break;


            double newRecon = this.squaredLoss();
            //prevent weights from exploding too far in either direction, we want this as close to zero as possible
            if(newRecon > currRecon || currRecon < 0 && newRecon < currRecon) {
                update((BaseNeuralNetwork) revert);
                log.info("Converged for new recon; breaking...");
                break;
            }
            else if(Double.isNaN(newRecon) || Double.isInfinite(newRecon)) {
                update((BaseNeuralNetwork) revert);
                log.info("Converged for new recon; breaking...");
                break;
            }


            else if(newRecon == currRecon)
                break;

            else {
                currRecon = newRecon;
                revert = clone();
                log.info("Recon went down " + currRecon);
            }

            iterations++;

            int plotIterations = getRenderIterations();
            if(plotIterations > 0) {
                NeuralNetPlotter plotter = new NeuralNetPlotter();
                if(iterations % plotIterations == 0) {
                    plotter.plotNetworkGradient(this,getGradient(extraParams),getInput().rows());
                }
            }

        }

    }

    public int getResetAdaGradIterations() {
        return resetAdaGradIterations;
    }

    public void setResetAdaGradIterations(int resetAdaGradIterations) {
        this.resetAdaGradIterations = resetAdaGradIterations;
    }

    public Map<Integer, Double> getMomentumAfter() {

        return momentumAfter;
    }

    public void setMomentumAfter(Map<Integer, Double> momentumAfter) {
        this.momentumAfter = momentumAfter;
    }

    /**
     * Whether to apply sparsity or not
     *
     * @return
     */
    @Override
    public boolean isApplySparsity() {
        return applySparsity;
    }

    @Override
    public boolean isUseAdaGrad() {
        return this.useAdaGrad;
    }


    @Override
    public boolean isUseRegularization() {
        return this.useRegularization;
    }

    @Override
    public void setUseRegularization(boolean useRegularization) {
        this.useRegularization = useRegularization;
    }
    /**
     * Applies sparsity to the passed in hbias gradient
     * @param hBiasGradient the hbias gradient to apply to
     * @param learningRate the learning rate used
     */
    protected void applySparsity(INDArray hBiasGradient,double learningRate) {

        if(useAdaGrad) {
            INDArray change = this.hBiasAdaGrad.getLearningRates(hBias).neg().mul(sparsity).mul(hBiasGradient.mul(sparsity));
            hBiasGradient.addi(change);
        }
        else {
            INDArray change = hBiasGradient.mul(sparsity).mul(-learningRate * sparsity);
            hBiasGradient.addi(change);

        }
    }

    @Override
    public boolean isConcatBiases() {
        return concatBiases;
    }

    @Override
    public void setConcatBiases(boolean concatBiases) {
        this.concatBiases = concatBiases;
    }

    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     * @param iteration the current iteration
     * @param learningRate the learning rate for the current iteration
     */
    protected void updateGradientAccordingToParams(NeuralNetworkGradient gradient,int iteration,double learningRate) {
        INDArray wGradient = gradient.getwGradient();

        INDArray hBiasGradient = gradient.gethBiasGradient();
        INDArray vBiasGradient = gradient.getvBiasGradient();

        //reset adagrad history
        if(iteration != 0 && resetAdaGradIterations > 0 &&  iteration % resetAdaGradIterations == 0) {
            wAdaGrad.historicalGradient = null;
            hBiasAdaGrad.historicalGradient = null;
            vBiasAdaGrad.historicalGradient = null;
            if(this.W != null && this.wAdaGrad == null)
                this.wAdaGrad = new AdaGrad(this.W.rows(),this.W.columns());

            if(this.vBias != null && this.vBiasAdaGrad == null)
                this.vBiasAdaGrad = new AdaGrad(this.vBias.rows(),this.vBias.columns());


            if(this.hBias != null && this.hBiasAdaGrad == null)
                this.hBiasAdaGrad = new AdaGrad(this.hBias.rows(),this.hBias.columns());

            log.info("Resetting adagrad");
        }

        INDArray wLearningRates = wAdaGrad.getLearningRates(wGradient);
        //change up momentum after so many iterations if specified
        double momentum = this.momentum;
        if(momentumAfter != null && !momentumAfter.isEmpty()) {
            int key = momentumAfter.keySet().iterator().next();
            if(iteration >= key) {
                momentum = momentumAfter.get(key);
            }
        }


        if (useAdaGrad)
            wGradient.muli(wLearningRates);
        else
            wGradient.muli(learningRate);

        if (useAdaGrad)
            hBiasGradient = hBiasGradient.mul(hBiasAdaGrad.getLearningRates(hBiasGradient));
        else
            hBiasGradient = hBiasGradient.mul(learningRate);


        if (useAdaGrad)
            vBiasGradient = vBiasGradient.mul(vBiasAdaGrad.getLearningRates(vBiasGradient));
        else
            vBiasGradient = vBiasGradient.mul(learningRate);



        //only do this with binary hidden layers
        if (applySparsity && this.hBiasGradient != null)
            applySparsity(hBiasGradient, learningRate);


        if (momentum != 0 && this.wGradient != null)
            wGradient.addi(this.wGradient.mul(momentum).add(wGradient.mul(1 - momentum)));


        if(momentum != 0 && this.vBiasGradient != null)
            vBiasGradient.addi(this.vBiasGradient.mul(momentum).add(vBiasGradient.mul(1 - momentum)));

        if(momentum != 0 && this.hBiasGradient != null)
            hBiasGradient.addi(this.hBiasGradient.mul(momentum).add(hBiasGradient.mul(1 - momentum)));




        if (normalizeByInputRows) {
            wGradient.divi(lastMiniBatchSize);
            vBiasGradient.divi(lastMiniBatchSize);
            hBiasGradient.divi(lastMiniBatchSize);
        }

        //simulate post gradient application  and apply the difference to the gradient to decrease the change the gradient has
        if(useRegularization && l2 > 0) {
            if(useAdaGrad)
                wGradient.subi(W.mul(l2).mul(wLearningRates));

            else
                wGradient.subi(W.mul(l2 * learningRate));

        }

        if(constrainGradientToUnitNorm) {
            wGradient.divi(wGradient.norm2(Integer.MAX_VALUE));
            vBiasGradient.divi(vBiasGradient.norm2(Integer.MAX_VALUE));
            hBiasGradient.divi(hBiasGradient.norm2(Integer.MAX_VALUE));
        }


        this.wGradient = wGradient;
        this.vBiasGradient = vBiasGradient;
        this.hBiasGradient = hBiasGradient;

    }



    /**
     * Clears the input from the neural net
     */
    @Override
    public void clearInput() {
        this.input = null;
    }

    @Override
    public void setDropOut(double dropOut) {
        this.dropOut = dropOut;
    }
    @Override
    public double dropOut() {
        return dropOut;
    }
    @Override
    public AdaGrad getAdaGrad() {
        return this.wAdaGrad;
    }
    @Override
    public void setAdaGrad(AdaGrad adaGrad) {
        this.wAdaGrad = adaGrad;
    }


    @Override
    public void setConstrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
        this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
    }



    /**
     * Whether to constrain the gradient to unit norm or not
     */
    @Override
    public boolean isConstrainGradientToUnitNorm() {
        return constrainGradientToUnitNorm;
    }

    @Override
    public NeuralNetwork transpose() {
        try {
            Constructor<?> c =  Dl4jReflection.getEmptyConstructor(getClass());
            c.setAccessible(true);
            NeuralNetwork ret = (NeuralNetwork) c.newInstance();
            ret.setMomentumAfter(momentumAfter);
            ret.setConcatBiases(concatBiases);
            ret.setResetAdaGradIterations(resetAdaGradIterations);
            ret.setVBiasAdaGrad(hBiasAdaGrad);
            ret.sethBias(vBias.dup());
            ret.setvBias(NDArrays.zeros(hBias.rows(),hBias.columns()));
            ret.setnHidden(getnVisible());
            ret.setnVisible(getnHidden());
            ret.setW(W.transpose());
            ret.setL2(l2);
            ret.setMomentum(momentum);
            ret.setRenderEpochs(getRenderIterations());
            ret.setSparsity(sparsity);
            ret.setRng(getRng());
            ret.setDist(getDist());
            ret.setAdaGrad(wAdaGrad);
            ret.setLossFunction(lossFunction);
            ret.setOptimizationAlgorithm(optimizationAlgo);
            ret.setConstrainGradientToUnitNorm(constrainGradientToUnitNorm);
            return ret;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }

    @Override
    public NeuralNetwork clone() {
        try {
            Constructor<?> c =  Dl4jReflection.getEmptyConstructor(getClass());
            c.setAccessible(true);
            NeuralNetwork ret = (NeuralNetwork) c.newInstance();
            ret.setMomentumAfter(momentumAfter);
            ret.setResetAdaGradIterations(resetAdaGradIterations);
            ret.setHbiasAdaGrad(hBiasAdaGrad);
            ret.setVBiasAdaGrad(vBiasAdaGrad);
            ret.sethBias(hBias.dup());
            ret.setvBias(vBias.dup());
            ret.setnHidden(getnHidden());
            ret.setnVisible(getnVisible());
            ret.setW(W.dup());
            ret.setL2(l2);
            ret.setMomentum(momentum);
            ret.setRenderEpochs(getRenderIterations());
            ret.setSparsity(sparsity);
            ret.setRng(getRng());
            ret.setDist(getDist());
            ret.setAdaGrad(wAdaGrad);
            ret.setLossFunction(lossFunction);
            ret.setConstrainGradientToUnitNorm(constrainGradientToUnitNorm);
            ret.setOptimizationAlgorithm(optimizationAlgo);
            return ret;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }

    /**
     * RMSE for reconstruction entropy
     *
     * @return rmse for reconstruction entropy
     */
    @Override
    public double mseRecon() {
        double recon = (double) sqrt(pow(reconstruct(input).subi(input),2)).sum(Integer.MAX_VALUE).element();
        return recon / input.rows();



    }

    @Override
    public LossFunction getLossFunction() {
        return lossFunction;
    }
    @Override
    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }
    @Override
    public OptimizationAlgorithm getOptimizationAlgorithm() {
        return optimizationAlgo;
    }
    @Override
    public void setOptimizationAlgorithm(
            OptimizationAlgorithm optimizationAlgorithm) {
        this.optimizationAlgo = optimizationAlgorithm;
    }
    @Override
    public RealDistribution getDist() {
        return dist;
    }

    @Override
    public void setDist(RealDistribution dist) {
        this.dist = dist;
    }

    @Override
    public void merge(NeuralNetwork network,int batchSize) {
        W.addi(network.getW().sub(W).div(batchSize));
        hBias.addi(network.gethBias().sub(hBias).divi(batchSize));
        vBias.addi(network.getvBias().subi(vBias).divi(batchSize));

    }


    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("BaseNeuralNetwork{");
        sb.append("nVisible=").append(nVisible);
        sb.append(", nHidden=").append(nHidden);
        sb.append(", rng=").append(rng);
        sb.append(", sparsity=").append(sparsity);
        sb.append(", momentum=").append(momentum);
        sb.append(", dist=").append(dist);
        sb.append(", l2=").append(l2);
        sb.append(", optimizer=").append(optimizer);
        sb.append(", renderWeightsEveryNumEpochs=").append(renderWeightsEveryNumEpochs);
        sb.append(", fanIn=").append(fanIn);
        sb.append(", useRegularization=").append(useRegularization);
        sb.append(", useAdaGrad=").append(useAdaGrad);
        sb.append(", firstTimeThrough=").append(firstTimeThrough);
        sb.append(", normalizeByInputRows=").append(normalizeByInputRows);
        sb.append(", applySparsity=").append(applySparsity);
        sb.append(", dropOut=").append(dropOut);
        sb.append(", doMask=").append(doMask);
        sb.append(", optimizationAlgo=").append(optimizationAlgo);
        sb.append(", lossFunction=").append(lossFunction);
        sb.append(", cacheInput=").append(cacheInput);
        sb.append(", wGradient=").append(wGradient);
        sb.append(", vBiasGradient=").append(vBiasGradient);
        sb.append(", hBiasGradient=").append(hBiasGradient);
        sb.append(", lastMiniBatchSize=").append(lastMiniBatchSize);
        sb.append(", wAdaGrad=").append(wAdaGrad);
        sb.append(", hBiasAdaGrad=").append(hBiasAdaGrad);
        sb.append(", vBiasAdaGrad=").append(vBiasAdaGrad);
        sb.append('}');
        return sb.toString();
    }

    /**
     * Copies params from the passed in network
     * to this one
     * @param n the network to copy
     */
    public void update(BaseNeuralNetwork n) {
        this.W = n.W;
        this.normalizeByInputRows = n.normalizeByInputRows;
        this.hBias = n.hBias;
        this.vBias = n.vBias;
        this.l2 = n.l2;
        this.dropOut = n.dropOut;
        this.applySparsity = n.applySparsity;
        this.momentumAfter = n.momentumAfter;
        this.useRegularization = n.useRegularization;
        this.momentum = n.momentum;
        this.nHidden = n.nHidden;
        this.nVisible = n.nVisible;
        this.rng = n.rng;
        this.sparsity = n.sparsity;
        this.wAdaGrad = n.wAdaGrad;
        this.hBiasAdaGrad = n.hBiasAdaGrad;
        this.vBiasAdaGrad = n.vBiasAdaGrad;
        this.optimizationAlgo = n.optimizationAlgo;
        this.lossFunction = n.lossFunction;
        this.cacheInput = n.cacheInput;
        this.constrainGradientToUnitNorm = n.constrainGradientToUnitNorm;
    }

    /**
     * Load (using {@link ObjectInputStream}
     * @param is the input stream to load from (usually a file)
     */
    public void load(InputStream is) {
        try {
            ObjectInputStream ois = new ObjectInputStream(is);
            BaseNeuralNetwork loaded = (BaseNeuralNetwork) ois.readObject();
            update(loaded);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }


    /**
     * Negative log likelihood of the current input given
     * the corruption level
     * @return the negative log likelihood of the auto encoder
     * given the corruption level
     */
    @Override
    public double negativeLogLikelihood() {
        INDArray z = this.reconstruct(input);
        if(this.useRegularization) {
            double reg = (2 / l2) * (double) pow(this.W,2).sum(Integer.MAX_VALUE).element();

            double ret = - (double) input.mul(log(z)).add(
                    input.rsub(1).muli(log(z.rsub(1)))).
                    sum(1).mean(Integer.MAX_VALUE).element() + reg;
            if(this.normalizeByInputRows)
                ret /= input.rows();
            return ret;
        }



        double likelihood =  - (double) input.mul(log(z)).add(
                input.rsub(1).muli(log(z.rsub(1)))).
                sum(1).mean(Integer.MAX_VALUE).element();

        if(this.normalizeByInputRows)
            likelihood /= input.rows();


        return likelihood;
    }


    /**
     * Negative log likelihood of the current input given
     * the corruption level
     * @return the negative log likelihood of the auto encoder
     * given the corruption level
     */
    public double negativeLoglikelihood(INDArray input) {
        INDArray z = this.reconstruct(input);
        if(this.useRegularization) {
            double reg = (2 / l2) * (double) pow(this.W,2).sum(Integer.MAX_VALUE).element();

            return - (double) input.mul(log(z)).add(
                    input.rsub(1).muli(log(z.rsub(1)))).
                    sum(1).mean(Integer.MAX_VALUE).element() + reg;
        }

        return - (double) input.mul(log(z)).add(
                input.rsub(1).muli(log(z.rsub(1)))).
                sum(1).mean(Integer.MAX_VALUE).element();
    }


    /**
     * Reconstruction entropy.
     * This compares the similarity of two probability
     * distributions, in this case that would be the input
     * and the reconstructed input with gaussian noise.
     * This will account for either regularization or none
     * depending on the configuration.
     * @return reconstruction error
     */
    public double getReConstructionCrossEntropy() {
       return org.deeplearning4j.linalg.lossfunctions.LossFunctions.reconEntropy(input,hBias,vBias,W);
    }


    @Override
    public boolean normalizeByInputRows() {
        return normalizeByInputRows;
    }
    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#getnVisible()
     */
    @Override
    public int getnVisible() {
        return nVisible;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#setnVisible(int)
     */
    @Override
    public void setnVisible(int nVisible) {
        this.nVisible = nVisible;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#getnHidden()
     */
    @Override
    public int getnHidden() {
        return nHidden;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#setnHidden(int)
     */
    @Override
    public void setnHidden(int nHidden) {
        this.nHidden = nHidden;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#getW()
     */
    @Override
    public INDArray getW() {
        return W;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#setW(org.jblas.INDArray)
     */
    @Override
    public void setW(INDArray w) {
        W = w;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#gethBias()
     */
    @Override
    public INDArray gethBias() {
        return hBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#sethBias(org.jblas.INDArray)
     */
    @Override
    public void sethBias(INDArray hBias) {
        this.hBias = hBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#getvBias()
     */
    @Override
    public INDArray getvBias() {
        return vBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#setvBias(org.jblas.INDArray)
     */
    @Override
    public void setvBias(INDArray vBias) {
        this.vBias = vBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#getRng()
     */
    @Override
    public RandomGenerator getRng() {
        return rng;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#setRng(org.apache.commons.math3.random.RandomGenerator)
     */
    @Override
    public void setRng(RandomGenerator rng) {
        this.rng = rng;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#getInput()
     */
    @Override
    public INDArray getInput() {
        return input;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#setInput(org.jblas.INDArray)
     */
    @Override
    public void setInput(INDArray input) {
        this.input = input;
    }


    public double getSparsity() {
        return sparsity;
    }
    public void setSparsity(double sparsity) {
        this.sparsity = sparsity;
    }
    public double getMomentum() {
        return momentum;
    }
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }
    public double getL2() {
        return l2;
    }
    public void setL2(double l2) {
        this.l2 = l2;
    }


    @Override
    public void setWeightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
    }

    @Override
    public WeightInit getWeightInit() {
        return weightInit;
    }

    @Override
    public AdaGrad gethBiasAdaGrad() {
        return hBiasAdaGrad;
    }
    @Override
    public void setHbiasAdaGrad(AdaGrad adaGrad) {
        this.hBiasAdaGrad = adaGrad;
    }
    @Override
    public AdaGrad getVBiasAdaGrad() {
        return this.vBiasAdaGrad;
    }
    @Override
    public void setVBiasAdaGrad(AdaGrad adaGrad) {
        this.vBiasAdaGrad = adaGrad;
    }
    /**
     * Write this to an object output stream
     * @param os the output stream to write to
     */
    public void write(OutputStream os) {
        try {
            ObjectOutputStream os2 = new ObjectOutputStream(os);
            os2.writeObject(this);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public INDArray output(INDArray x) {
        return reconstruct(x);
    }

    /**
     * All neural networks are based on this idea of
     * minimizing reconstruction error.
     * Both RBMs and Denoising AutoEncoders
     * have a component for reconstructing, ala different implementations.
     *
     * @param x the input to reconstruct
     * @return the reconstructed input
     */
    public abstract INDArray reconstruct(INDArray x);

    /**
     * The loss function (cross entropy, reconstruction error,...)
     * @return the loss function
     */
    public abstract double lossFunction(Object[] params);


    public double lossFunction() {
        return lossFunction(null);
    }



    protected void applyDropOutIfNecessary(INDArray input) {
        if(dropOut > 0) {
            this.doMask = NDArrays.rand(input.rows(), input.columns()).gt(dropOut);
        }

        else
            this.doMask = NDArrays.ones(input.rows(),input.columns());

        //actually apply drop out
        input.muli(doMask);

    }

    /**
     * train one iteration of the network
     * @param input the input to train on
     * @param lr the learning rate to train at
     * @param params the extra params (k, corruption level,...)
     */
    @Override
    public abstract void train(INDArray input,double lr,Object[] params);

    @Override
    public double squaredLoss() {
        INDArray squaredDiff = pow(reconstruct(input).sub(input),2);
        double loss = (double) squaredDiff.sum(Integer.MAX_VALUE).element() / input.rows();
        if(this.useRegularization) {
            loss += 0.5 * l2 * (double) pow(W,2).sum(Integer.MAX_VALUE).element();
        }

        return loss;
    }


    @Override
    public double mse() {
        INDArray reconstructed = reconstruct(input);
        INDArray diff = reconstructed.sub(input);
        double sum = 0.5 * (double) pow(diff,2).sum(1).sum(Integer.MAX_VALUE).element() / input.rows();
        return sum;
    }

    @Override
    public INDArray hBiasMean() {
        INDArray hbiasMean = getInput().mmul(getW()).addRowVector(gethBias());
        return hbiasMean;
    }

    //align input so it can be used in training
    protected INDArray preProcessInput(INDArray input) {
        if(concatBiases)
            return NDArrays.concatHorizontally(input,NDArrays.ones(input.rows(),1));
        return input;
    }

    @Override
    public void iterationDone(int epoch) {
        int plotEpochs = getRenderIterations();
        if(plotEpochs <= 0)
            return;
        if(epoch % plotEpochs == 0 || epoch == 0) {
            NeuralNetPlotter plotter = new NeuralNetPlotter();
            plotter.plotNetworkGradient(this,this.getGradient(new Object[]{1,0.001,1000}),getInput().rows());
        }
    }
    public static class Builder<E extends BaseNeuralNetwork> {
        private E ret = null;
        private INDArray W;
        protected Class<? extends NeuralNetwork> clazz;
        private INDArray vBias;
        private INDArray hBias;
        private int numVisible;
        private int numHidden;
        private RandomGenerator gen = new MersenneTwister(123);
        private INDArray input;
        private double sparsity = 0;
        private double l2 = 0.01;
        private double momentum = 0.5;
        private int renderWeightsEveryNumEpochs = -1;
        private double fanIn = 0.1;
        private boolean useRegularization = false;
        private RealDistribution dist;
        private boolean useAdaGrad = true;
        private boolean applySparsity = false;
        private boolean normalizeByInputRows = true;
        private double dropOut = 0;
        private LossFunction lossFunction = LossFunction.RECONSTRUCTION_CROSSENTROPY;
        private OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT;
        private boolean cacheInput = true;
        //momentum after n iterations
        protected Map<Integer,Double> momentumAfter = new HashMap<>();
        //reset adagrad historical gradient after n iterations
        protected int resetAdaGradIterations = -1;
        protected boolean concatBiases = false;
        private boolean constrainGradientToUnitNorm = false;
        //private init scheme, this can either be a distribution or a applyTransformToDestination scheme
        protected WeightInit weightInit;

        /**
         * Weight initialization scheme
         * @param weightInit the weight initialization scheme
         * @return
         */
        public Builder<E> weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        /**
         * Constrains gradient to unit norm when updating parameters
         * @param constrainGradientToUnitNorm whether to constrain the gradient to unit norm or not
         * @return
         */
        public Builder<E> constrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
            this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
            return this;
        }

        /**
         * Whether to concat biases or add them on the neural net
         * @param concatBiases
         * @return
         */
        public Builder<E> concatBiases(boolean concatBiases) {
            this.concatBiases = concatBiases;
            return this;
        }

        public Builder<E> resetAdaGradIterations(int resetAdaGradIterations) {
            this.resetAdaGradIterations = resetAdaGradIterations;
            return this;
        }

        public Builder<E> momentumAfter(Map<Integer,Double> momentumAfter) {
            this.momentumAfter = momentumAfter;
            return this;
        }

        public Builder<E> cacheInput(boolean cacheInput) {
            this.cacheInput = cacheInput;
            return this;
        }

        public Builder<E> applySparsity(boolean applySparsity) {
            this.applySparsity = applySparsity;
            return this;
        }


        public Builder<E> withOptmizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        public Builder<E> withLossFunction(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }


        public Builder<E> withDropOut(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }
        public Builder<E> normalizeByInputRows(boolean normalizeByInputRows) {
            this.normalizeByInputRows = normalizeByInputRows;
            return this;
        }

        public Builder<E> useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder<E> withDistribution(RealDistribution dist) {
            this.dist = dist;
            return this;
        }

        public Builder<E> useRegularization(boolean useRegularization) {
            this.useRegularization = useRegularization;
            return this;
        }

        public Builder<E> fanIn(double fanIn) {
            this.fanIn = fanIn;
            return this;
        }

        public Builder<E> withL2(double l2) {
            this.l2 = l2;
            return this;
        }


        public Builder<E> renderWeights(int numEpochs) {
            this.renderWeightsEveryNumEpochs = numEpochs;
            return this;
        }

        @SuppressWarnings("unchecked")
        public E buildEmpty() {
            try {
                return (E) clazz.newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                throw new RuntimeException(e);
            }
        }



        public Builder<E> withClazz(Class<? extends BaseNeuralNetwork> clazz) {
            this.clazz = clazz;
            return this;
        }

        public Builder<E> withSparsity(double sparsity) {
            this.sparsity = sparsity;
            return this;
        }
        public Builder<E> withMomentum(double momentum) {
            this.momentum = momentum;
            return this;
        }

        public Builder<E> withInput(INDArray input) {
            this.input = input;
            return this;
        }

        public Builder<E> asType(Class<E> clazz) {
            this.clazz = clazz;
            return this;
        }


        public Builder<E> withWeights(INDArray W) {
            this.W = W;
            return this;
        }

        public Builder<E> withVisibleBias(INDArray vBias) {
            this.vBias = vBias;
            return this;
        }

        public Builder<E> withHBias(INDArray hBias) {
            this.hBias = hBias;
            return this;
        }

        public Builder<E> numberOfVisible(int numVisible) {
            this.numVisible = numVisible;
            return this;
        }

        public Builder<E> numHidden(int numHidden) {
            this.numHidden = numHidden;
            return this;
        }

        public Builder<E> withRandom(RandomGenerator gen) {
            this.gen = gen;
            return this;
        }

        public E build() {
            return buildWithInput();

        }


        @SuppressWarnings("unchecked")
        private  E buildWithInput()  {
            Constructor<?>[] c = clazz.getDeclaredConstructors();
            for(int i = 0; i < c.length; i++) {
                Constructor<?> curr = c[i];
                curr.setAccessible(true);
                Class<?>[] classes = curr.getParameterTypes();
                //input matrix found
                if(classes != null && classes.length > 0 && classes[0].isAssignableFrom(INDArray.class)) {
                    try {
                        ret = (E) curr.newInstance(input,numVisible, numHidden, W, hBias,vBias, gen,fanIn,dist);
                        ret.cacheInput = cacheInput;
                        ret.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
                        ret.weightInit = weightInit;
                        ret.concatBiases = concatBiases;
                        ret.sparsity = this.sparsity;
                        ret.resetAdaGradIterations = resetAdaGradIterations;
                        ret.momentumAfter = momentumAfter;
                        ret.applySparsity = this.applySparsity;
                        ret.normalizeByInputRows = this.normalizeByInputRows;
                        ret.renderWeightsEveryNumEpochs = this.renderWeightsEveryNumEpochs;
                        ret.l2 = this.l2;
                        ret.momentum = this.momentum;
                        ret.useRegularization = this.useRegularization;
                        ret.useAdaGrad = this.useAdaGrad;
                        ret.dropOut = this.dropOut;
                        ret.optimizationAlgo = this.optimizationAlgo;
                        ret.lossFunction = this.lossFunction;
                        return ret;
                    }catch(Exception e) {
                        throw new RuntimeException(e);
                    }

                }
            }
            return ret;
        }
    }



}
