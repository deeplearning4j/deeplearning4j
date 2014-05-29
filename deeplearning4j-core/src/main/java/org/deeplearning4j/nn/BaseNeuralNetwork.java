package org.deeplearning4j.nn;

import static org.deeplearning4j.util.MatrixUtil.log;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.deeplearning4j.util.MatrixUtil.sqrt;
import static org.jblas.MatrixFunctions.pow;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Constructor;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.optimize.NeuralNetworkOptimizer;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.util.Dl4jReflection;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
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
    protected DoubleMatrix W;
    /* hidden bias */
    protected DoubleMatrix hBias;
    /* visible bias */
    protected DoubleMatrix vBias;
    /* RNG for sampling. */
    protected RandomGenerator rng;
    /* input to the network */
    protected DoubleMatrix input;
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
    protected DoubleMatrix doMask;
    protected OptimizationAlgorithm optimizationAlgo;
    protected LossFunction lossFunction;
    private static Logger log = LoggerFactory.getLogger(BaseNeuralNetwork.class);
    //cache input when training?
    protected boolean cacheInput;
    //previous gradient used for updates
    protected DoubleMatrix wGradient,vBiasGradient,hBiasGradient;

    protected int lastMiniBatchSize = 1;

    protected AdaGrad wAdaGrad,hBiasAdaGrad,vBiasAdaGrad;


    protected BaseNeuralNetwork() {}
    /**
     *
     * @param nVisible the number of outbound nodes
     * @param nHidden the number of nodes in the hidden layer
     * @param W the weights for this vector, maybe null, if so this will
     * create a matrix with nHidden x nVisible dimensions.
     * @param rng the rng, if not a seed of 1234 is used.
     */
    public BaseNeuralNetwork(int nVisible, int nHidden,
                             DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng,double fanIn,RealDistribution dist) {
        this(null,nVisible,nHidden,W,hbias,vbias,rng,fanIn,dist);

    }

    /**
     *
     * @param input the input examples
     * @param nVisible the number of outbound nodes
     * @param nHidden the number of nodes in the hidden layer
     * @param W the weights for this vector, maybe null, if so this will
     * create a matrix with nHidden x nVisible dimensions.
     * @param rng the rng, if not a seed of 1234 is used.
     */
    public BaseNeuralNetwork(DoubleMatrix input, int nVisible, int nHidden,
                             DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng,double fanIn,RealDistribution dist) {
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
            this.wAdaGrad = new AdaGrad(this.W.rows,this.W.columns);

        this.vBias = vbias;
        if(this.vBias != null)
            this.vBiasAdaGrad = new AdaGrad(this.vBias.rows,this.vBias.columns);


        this.hBias = hbias;
        if(this.hBias != null)
            this.hBiasAdaGrad = new AdaGrad(this.hBias.rows,this.hBias.columns);


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
        return (MatrixFunctions.pow(getW(),2).sum()/ 2.0)  * l2 + 1e-6;
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

            this.W = DoubleMatrix.zeros(nVisible,nHidden);

            for(int i = 0; i < this.W.rows; i++)
                this.W.putRow(i,new DoubleMatrix(dist.sample(this.W.columns)));

        }

        this.wAdaGrad = new AdaGrad(this.W.rows,this.W.columns);

        if(this.hBias == null) {
            this.hBias = DoubleMatrix.zeros(nHidden);
			/*
			 * Encourage sparsity.
			 * See Hinton's Practical guide to RBMs
			 */
            //this.hBias.subi(4);
        }

        this.hBiasAdaGrad = new AdaGrad(hBias.rows,hBias.columns);


        if(this.vBias == null) {
            if(this.input != null) {

                this.vBias = DoubleMatrix.zeros(nVisible);


            }
            else
                this.vBias = DoubleMatrix.zeros(nVisible);
        }

        this.vBiasAdaGrad = new AdaGrad(vBias.rows,vBias.columns);


    }




    @Override
    public void resetAdaGrad(double lr) {
        if(!firstTimeThrough) {
            this.wAdaGrad = new AdaGrad(this.getW().rows,this.getW().columns,lr);
            firstTimeThrough = false;
        }

    }

    public void setRenderEpochs(int renderEpochs) {
        this.renderWeightsEveryNumEpochs = renderEpochs;

    }
    @Override
    public int getRenderEpochs() {
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
    public void backProp(double lr,int epochs,Object[] extraParams) {
        double currRecon = getReConstructionCrossEntropy();
        boolean train = true;
        NeuralNetwork revert = clone();
        int numEpochs = 0;
        while(train) {
            if(numEpochs > epochs)
                break;

            NeuralNetworkGradient gradient = getGradient(extraParams);
            DoubleMatrix wLearningRates = getAdaGrad().getLearningRates(gradient.getwGradient());
            Pair<DoubleMatrix,DoubleMatrix> sample = sampleHiddenGivenVisible(input);
            DoubleMatrix hiddenSample = sample.getSecond().transpose();
            /*
            Scale the input and reconstrution to see the relative difference in absolute space
             */
            DoubleMatrix scaledInput = input.dup();
            MatrixUtil.normalizeZeroMeanAndUnitVariance(scaledInput);
            DoubleMatrix z = reconstruct(input);
            MatrixUtil.normalizeZeroMeanAndUnitVariance(z);
            DoubleMatrix outputDiff = z.sub(scaledInput);
            //changes in error relative to neurons
            DoubleMatrix delta = hiddenSample.mmul(outputDiff).transpose();
            //hidden activations
            DoubleMatrix hBiasMean = sample.getFirst().columnSums().transpose();

            if(isUseAdaGrad())
                delta.muli(wLearningRates);
            else
                delta.muli(lr);

            if(momentum != 0)
                delta.muli(momentum).add(delta.mul(1 - momentum));

            if(normalizeByInputRows)
                delta.divi(input.rows);


            getW().addi(W.sub(delta));

            if(isUseAdaGrad())
                hBiasMean.muli(gethBiasAdaGrad().getLearningRates(gradient.gethBiasGradient()));
            else
                hBiasMean.muli(lr);

            if(momentum != 0)
                hBiasMean.muli(momentum).add(hBiasMean.mul(1 - momentum));

            if(normalizeByInputRows)
                hBiasMean.divi(input.rows);

            gethBias().addi(gethBias().sub(hBiasMean));

            double newRecon = this.getReConstructionCrossEntropy();
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

            numEpochs++;

            int plotEpochs = getRenderEpochs();
            if(plotEpochs > 0) {
                NeuralNetPlotter plotter = new NeuralNetPlotter();
                if(numEpochs % plotEpochs == 0) {
                    plotter.plotNetworkGradient(this,getGradient(extraParams));
                }
            }

        }

    }

    @Override
    public boolean isUseAdaGrad() {
        return this.useAdaGrad;
    }


    @Override
    public boolean isUseRegularization() {
        return this.useRegularization;
    }

    /**
     * Applies sparsity to the passed in hbias gradient
     * @param hBiasGradient the hbias gradient to apply to
     * @param learningRate the learning rate used
     */
    protected void applySparsity(DoubleMatrix hBiasGradient,double learningRate) {

        if(useAdaGrad) {
            DoubleMatrix change = this.hBiasAdaGrad.getLearningRates(hBias).neg().mul(sparsity).mul(hBiasGradient.mul(sparsity));
            hBiasGradient.addi(change);
        }
        else {
            DoubleMatrix change = hBiasGradient.mul(sparsity).mul(-learningRate * sparsity);
            hBiasGradient.addi(change);

        }
    }




    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     * @param learningRate the learning rate for the current iteratiaon
     */
    protected void updateGradientAccordingToParams(NeuralNetworkGradient gradient,double learningRate) {
        DoubleMatrix wGradient = gradient.getwGradient();

        DoubleMatrix hBiasGradient = gradient.gethBiasGradient();
        DoubleMatrix vBiasGradient = gradient.getvBiasGradient();
        DoubleMatrix wLearningRates = wAdaGrad.getLearningRates(wGradient);
        if (useAdaGrad)
            wGradient.muli(wLearningRates);
        else
            wGradient.muli(learningRate);

        if (useAdaGrad)
            hBiasGradient = hBiasGradient.mul(hBiasAdaGrad.getLearningRates(hBiasGradient)).add(hBiasGradient.mul(momentum));
        else
            hBiasGradient = hBiasGradient.mul(learningRate).add(hBiasGradient.mul(momentum));


        if (useAdaGrad)
            vBiasGradient = vBiasGradient.mul(vBiasAdaGrad.getLearningRates(vBiasGradient)).add(vBiasGradient.mul(momentum));
        else
            vBiasGradient = vBiasGradient.mul(learningRate).add(vBiasGradient.mul(momentum));



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
        if(useRegularization && l2 > 0)
            wGradient.subi(wGradient.mul(l2).mul(wLearningRates));

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
    public NeuralNetwork transpose() {
        try {
            Constructor<?> c =  Dl4jReflection.getEmptyConstructor(getClass());
            c.setAccessible(true);
            NeuralNetwork ret = (NeuralNetwork) c.newInstance();
            ret.setHbiasAdaGrad(vBiasAdaGrad);
            ret.setVBiasAdaGrad(hBiasAdaGrad);
            ret.sethBias(vBias.dup());
            ret.setvBias(hBias.dup());
            ret.setnHidden(getnVisible());
            ret.setnVisible(getnHidden());
            ret.setW(W.transpose());
            ret.setL2(l2);
            ret.setMomentum(momentum);
            ret.setRenderEpochs(getRenderEpochs());
            ret.setSparsity(sparsity);
            ret.setRng(getRng());
            ret.setDist(getDist());
            ret.setAdaGrad(wAdaGrad);
            ret.setLossFunction(lossFunction);
            ret.setOptimizationAlgorithm(optimizationAlgo);
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
            ret.setHbiasAdaGrad(hBiasAdaGrad);
            ret.setVBiasAdaGrad(vBiasAdaGrad);
            ret.sethBias(hBias.dup());
            ret.setvBias(vBias.dup());
            ret.setnHidden(getnHidden());
            ret.setnVisible(getnVisible());
            ret.setW(W.dup());
            ret.setL2(l2);
            ret.setMomentum(momentum);
            ret.setRenderEpochs(getRenderEpochs());
            ret.setSparsity(sparsity);
            ret.setRng(getRng());
            ret.setDist(getDist());
            ret.setAdaGrad(wAdaGrad);
            ret.setLossFunction(lossFunction);
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
        return sqrt(pow(reconstruct(input).sub(input),2)).sum() / input.rows;



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
        DoubleMatrix z = this.reconstruct(input);
        if(this.useRegularization) {
            double reg = (2 / l2) * MatrixFunctions.pow(this.W,2).sum();

            double ret = - input.mul(log(z)).add(
                    oneMinus(input).mul(log(oneMinus(z)))).
                    columnSums().mean() + reg;
            if(this.normalizeByInputRows)
                ret /= input.rows;
            return ret;
        }



        double likelihood =  - input.mul(log(z)).add(
                oneMinus(input).mul(log(oneMinus(z)))).
                columnSums().mean();

        if(this.normalizeByInputRows)
            likelihood /= input.rows;


        return likelihood;
    }


    /**
     * Negative log likelihood of the current input given
     * the corruption level
     * @return the negative log likelihood of the auto encoder
     * given the corruption level
     */
    public double negativeLoglikelihood(DoubleMatrix input) {
        DoubleMatrix z = this.reconstruct(input);
        if(this.useRegularization) {
            double reg = (2 / l2) * MatrixFunctions.pow(this.W,2).sum();

            return - input.mul(log(z)).add(
                    oneMinus(input).mul(log(oneMinus(z)))).
                    columnSums().mean() + reg;
        }

        return - input.mul(log(z)).add(
                oneMinus(input).mul(log(oneMinus(z)))).
                columnSums().mean();
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
        DoubleMatrix preSigH = input.mmul(W).addRowVector(hBias);
        DoubleMatrix sigH = sigmoid(preSigH);

        DoubleMatrix preSigV = sigH.mmul(W.transpose()).addRowVector(vBias);
        DoubleMatrix sigV = sigmoid(preSigV);
        DoubleMatrix inner =
                input.mul(log(sigV))
                        .add(oneMinus(input)
                                .mul(log(oneMinus(sigV))));

        double ret =   inner.rowSums().mean();
        if(normalizeByInputRows)
            ret /= input.rows;

        return ret;
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
    public DoubleMatrix getW() {
        return W;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#setW(org.jblas.DoubleMatrix)
     */
    @Override
    public void setW(DoubleMatrix w) {
        W = w;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#gethBias()
     */
    @Override
    public DoubleMatrix gethBias() {
        return hBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#sethBias(org.jblas.DoubleMatrix)
     */
    @Override
    public void sethBias(DoubleMatrix hBias) {
        this.hBias = hBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#getvBias()
     */
    @Override
    public DoubleMatrix getvBias() {
        return vBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#setvBias(org.jblas.DoubleMatrix)
     */
    @Override
    public void setvBias(DoubleMatrix vBias) {
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
    public DoubleMatrix getInput() {
        return input;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.NeuralNetwork#setInput(org.jblas.DoubleMatrix)
     */
    @Override
    public void setInput(DoubleMatrix input) {
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

    /**
     * All neural networks are based on this idea of
     * minimizing reconstruction error.
     * Both RBMs and Denoising AutoEncoders
     * have a component for reconstructing, ala different implementations.
     *
     * @param x the input to reconstruct
     * @return the reconstructed input
     */
    public abstract DoubleMatrix reconstruct(DoubleMatrix x);

    /**
     * The loss function (cross entropy, reconstruction error,...)
     * @return the loss function
     */
    public abstract double lossFunction(Object[] params);


    public double lossFunction() {
        return lossFunction(null);
    }



    protected void applyDropOutIfNecessary(DoubleMatrix input) {
        if(dropOut > 0)
            this.doMask = DoubleMatrix.rand(input.rows, this.nHidden).gt(dropOut);

        else
            this.doMask = DoubleMatrix.ones(input.rows,this.nHidden);
    }

    /**
     * train one iteration of the network
     * @param input the input to train on
     * @param lr the learning rate to train at
     * @param params the extra params (k, corruption level,...)
     */
    @Override
    public abstract void train(DoubleMatrix input,double lr,Object[] params);

    @Override
    public double squaredLoss() {
        DoubleMatrix reconstructed = reconstruct(input);
        double loss = MatrixFunctions.powi(reconstructed.sub(input), 2).sum() / input.rows;
        if(this.useRegularization) {
            loss += 0.5 * l2 * MatrixFunctions.pow(W,2).sum();
        }

        return -loss;
    }


    @Override
    public double mse() {
        DoubleMatrix reconstructed = reconstruct(input);
        DoubleMatrix diff = reconstructed.sub(input);
        double sum = 0.5 * MatrixFunctions.pow(diff,2).columnSums().sum() / input.rows;
        return -sum;
    }

    @Override
    public DoubleMatrix hBiasMean() {
        DoubleMatrix hbiasMean = getInput().mmul(getW()).addRowVector(gethBias());
        return hbiasMean;
    }
    @Override
    public void epochDone(int epoch) {
        int plotEpochs = getRenderEpochs();
        if(plotEpochs <= 0)
            return;
        if(epoch % plotEpochs == 0 || epoch == 0) {
            NeuralNetPlotter plotter = new NeuralNetPlotter();
            plotter.plotNetworkGradient(this,this.getGradient(new Object[]{1,0.001,1000}));
        }
    }



    public static class Builder<E extends BaseNeuralNetwork> {
        private E ret = null;
        private DoubleMatrix W;
        protected Class<? extends NeuralNetwork> clazz;
        private DoubleMatrix vBias;
        private DoubleMatrix hBias;
        private int numVisible;
        private int numHidden;
        private RandomGenerator gen = new MersenneTwister(123);
        private DoubleMatrix input;
        private double sparsity = 0.01;
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

        public Builder<E> withInput(DoubleMatrix input) {
            this.input = input;
            return this;
        }

        public Builder<E> asType(Class<E> clazz) {
            this.clazz = clazz;
            return this;
        }


        public Builder<E> withWeights(DoubleMatrix W) {
            this.W = W;
            return this;
        }

        public Builder<E> withVisibleBias(DoubleMatrix vBias) {
            this.vBias = vBias;
            return this;
        }

        public Builder<E> withHBias(DoubleMatrix hBias) {
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
                if(classes != null && classes.length > 0 && classes[0].isAssignableFrom(DoubleMatrix.class)) {
                    try {
                        ret = (E) curr.newInstance(input,numVisible, numHidden, W, hBias,vBias, gen,fanIn,dist);
                        ret.cacheInput = cacheInput;
                        ret.sparsity = this.sparsity;
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
