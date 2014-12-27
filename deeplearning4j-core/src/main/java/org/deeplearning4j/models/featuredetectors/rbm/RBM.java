package org.deeplearning4j.models.featuredetectors.rbm;


import static org.nd4j.linalg.ops.transforms.Transforms.*;


import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.sampling.Sampling;
import org.deeplearning4j.nn.BasePretrainNetwork;
import org.deeplearning4j.util.RBMUtil;




/**
 * Restricted Boltzmann Machine.
 *
 * Markov chain with gibbs sampling.
 *
 * Supports the following visible units:
 *
 *     binary
 *     gaussian
 *     softmax
 *     linear
 *
 * Supports the following hidden units:
 *     rectified
 *     binary
 *     gaussian
 *     softmax
 *     linear
 *
 * Based on Hinton et al.'s work
 *
 * Great reference:
 * http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/239
 *
 *
 * @author Adam Gibson
 *
 */
public  class RBM extends BasePretrainNetwork {


    public RBM(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    public  static enum VisibleUnit {
        BINARY,GAUSSIAN,SOFTMAX,LINEAR


    }

    public  static enum HiddenUnit {
        RECTIFIED,BINARY,GAUSSIAN,SOFTMAX
    }
    /**
     *
     */
    private static final long serialVersionUID = 6189188205731511957L;
    protected INDArray sigma,hiddenSigma;






    /**
     * Contrastive divergence revolves around the idea
     * of approximating the log likelihood around x1(input) with repeated sampling.
     * Given is an energy based model: the higher k is (the more we sample the model)
     * the more we lower the energy (increase the likelihood of the model)
     *
     * and lower the likelihood (increase the energy) of the hidden samples.
     *
     * Other insights:
     *    CD - k involves keeping the first k samples of a gibbs sampling of the model.

     */
    public void contrastiveDivergence() {
        Gradient gradient = getGradient();
        getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY).subi(gradient.gradientLookupTable().get(PretrainParamInitializer.VISIBLE_BIAS_KEY));
        getParam(PretrainParamInitializer.BIAS_KEY).subi(gradient.gradientLookupTable().get(PretrainParamInitializer.BIAS_KEY));
        getParam(PretrainParamInitializer.WEIGHT_KEY).subi(gradient.gradientLookupTable().get(PretrainParamInitializer.WEIGHT_KEY));


    }


    @Override
    public Gradient getGradient() {



        int k = conf.getK();
        double learningRate = conf.getLr();


		/*
		 * Cost and updates dictionary.
		 * This is the update rules for weights and biases
		 */

        //POSITIVE PHASE
        Pair<INDArray,INDArray> probHidden = sampleHiddenGivenVisible(input);

		/*
		 * Start the gibbs sampling.
		 */
        INDArray chainStart = probHidden.getSecond();

		/*
		 * Note that at a later date, we can explore alternative methods of 
		 * storing the chain transitions for different kinds of sampling
		 * and exploring the search space.
		 */
        Pair<Pair<INDArray,INDArray>,Pair<INDArray,INDArray>> matrices;
        //negative visible means or expected values
        INDArray nvMeans = null;
        //negative value samples
        INDArray nvSamples = null;
        //negative hidden means or expected values
        INDArray nhMeans = null;
        //negative hidden samples
        INDArray nhSamples = null;

		/*
		 * K steps of gibbs sampling. This is the positive phase of contrastive divergence.
		 * 
		 * There are 4 matrices being computed for each gibbs sampling.
		 * The samples from both the positive and negative phases and their expected values 
		 * or averages.
		 * 
		 */

        for(int i = 0; i < k; i++) {

            //NEGATIVE PHASE
            if(i == 0)
                matrices = gibbhVh(chainStart);
            else
                matrices = gibbhVh(nhSamples);

            //getFromOrigin the cost updates for sampling in the chain after k iterations
            nvMeans = matrices.getFirst().getFirst();
            nvSamples = matrices.getFirst().getSecond();
            nhMeans = matrices.getSecond().getFirst();
            nhSamples = matrices.getSecond().getSecond();
        }

		/*
		 * Update gradient parameters
		 */
        INDArray wGradient = input.transpose().mmul(probHidden.getSecond()).sub(
                nvSamples.transpose().mmul(nhMeans)
        );



        INDArray hBiasGradient;

        if(conf.getSparsity() != 0)
            //all hidden units must stay around this number
            hBiasGradient = probHidden.getSecond().rsubi(conf.getSparsity()).mean(0);
        else
            //update rule: the expected values of the hidden input - the negative hidden  means adjusted by the learning rate
            hBiasGradient = probHidden.getSecond().sub(nhMeans).mean(0);




        //update rule: the expected values of the input - the negative samples adjusted by the learning rate
        INDArray  vBiasGradient = input.sub(nvSamples).mean(0);
        Gradient ret = new DefaultGradient();
        ret.gradientLookupTable().put(PretrainParamInitializer.VISIBLE_BIAS_KEY,vBiasGradient);
        ret.gradientLookupTable().put(PretrainParamInitializer.BIAS_KEY,hBiasGradient);
        ret.gradientLookupTable().put(PretrainParamInitializer.WEIGHT_KEY,wGradient);

        return ret;
    }



    @Override
    public Layer transpose() {
        RBM r = (RBM) super.transpose();
        HiddenUnit h = RBMUtil.inverse(conf.getVisibleUnit());
        VisibleUnit v = RBMUtil.inverse(conf.getHiddenUnit());
        if(h == null)
            h = conf.getHiddenUnit();
        if(v == null)
            v = conf.getVisibleUnit();

        r.sigma = sigma;
        r.hiddenSigma = hiddenSigma;
        return r;
    }



    /**
     * Free energy for an RBM
     * Lower energy models have higher probability
     * of activations
     * @param visibleSample the sample to test on
     * @return the free energy for this sample
     */
    public double freeEnergy(INDArray visibleSample) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray hBias = getParam(PretrainParamInitializer.BIAS_KEY);
        INDArray vBias = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);

        INDArray wxB = visibleSample.mmul(W).addiRowVector(hBias);
        double vBiasTerm = Nd4j.getBlasWrapper().dot(visibleSample, vBias);
        double hBiasTerm = log(exp(wxB).add(1)).sum(Integer.MAX_VALUE).getDouble(0);
        return -hBiasTerm - vBiasTerm;
    }


    /**
     * Binomial sampling of the hidden values given visible
     * @param v the visible values
     * @return a binomial distribution containing the expected values and the samples
     */
    @Override
    public Pair<INDArray,INDArray> sampleHiddenGivenVisible(INDArray v) {
        if(conf.getHiddenUnit() == HiddenUnit.RECTIFIED) {
            INDArray h1Mean = propUp(v);
            INDArray sigH1Mean = sigmoid(h1Mean);
            RandomGenerator gen = new MersenneTwister(123);
		/*
		 * Rectified linear part
		 */
            INDArray sqrtSigH1Mean = sqrt(sigH1Mean);
            //NANs here with Word2Vec
            INDArray sample = Sampling.normal(gen, h1Mean,1);
            sample.muli(sqrtSigH1Mean);
            INDArray h1Sample = h1Mean.add(sample);
            h1Sample = Transforms.max(h1Sample);
            //apply dropout
            applyDropOutIfNecessary(h1Sample);


            return new Pair<>(h1Mean,h1Sample);

        }

        else if(conf.getHiddenUnit() == HiddenUnit.GAUSSIAN) {
            INDArray h1Mean = propUp(v);
            this.hiddenSigma = h1Mean.var(1);

            INDArray h1Sample =  h1Mean.addi(Sampling.normal(conf.getRng(),h1Mean,this.hiddenSigma));

            //apply dropout
            applyDropOutIfNecessary(h1Sample);
            return new Pair<>(h1Mean,h1Sample);
        }

        else if(conf.getHiddenUnit() == HiddenUnit.SOFTMAX) {
            INDArray h1Mean = propUp(v);
            INDArray h1Sample = Activations.softMaxRows().apply(h1Mean);
            applyDropOutIfNecessary(h1Sample);
            return new Pair<>(h1Mean,h1Sample);
        }



        else if(conf.getHiddenUnit() == HiddenUnit.BINARY) {
            INDArray h1Mean = propUp(v);
            INDArray h1Sample = Sampling.binomial(h1Mean, 1, conf.getRng());
            applyDropOutIfNecessary(h1Sample);
            return new Pair<>(h1Mean,h1Sample);
        }



        throw new IllegalStateException("Hidden unit type must either be rectified linear or binary");

    }

    /**
     * Gibbs sampling step: hidden ---> visible ---> hidden
     * @param h the hidden input
     * @return the expected values and samples of both the visible samples given the hidden
     * and the new hidden input and expected values
     */
    public Pair<Pair<INDArray,INDArray>,Pair<INDArray,INDArray>> gibbhVh(INDArray h) {

        Pair<INDArray,INDArray> v1MeanAndSample = sampleVisibleGivenHidden(h);
        INDArray vSample = v1MeanAndSample.getSecond();

        Pair<INDArray,INDArray> h1MeanAndSample = sampleHiddenGivenVisible(vSample);
        return new Pair<>(v1MeanAndSample,h1MeanAndSample);
    }


    /**
     * Guess the visible values given the hidden
     * @param h the hidden units
     * @return a visible mean and sample relative to the hidden states
     * passed in
     */
    @Override
    public Pair<INDArray,INDArray> sampleVisibleGivenHidden(INDArray h) {
        INDArray v1Mean = propDown(h);

        if(conf.getVisibleUnit() == VisibleUnit.GAUSSIAN) {
            INDArray v1Sample = v1Mean.add(Nd4j.randn(v1Mean.rows(),v1Mean.columns(),conf.getRng()));
            return new Pair<>(v1Mean,v1Sample);

        }

        else if(conf.getVisibleUnit() == VisibleUnit.LINEAR) {
            INDArray v1Sample = Sampling.normal(conf.getRng(),v1Mean,1);
            return new Pair<>(v1Mean,v1Sample);
        }

        else if(conf.getVisibleUnit() == VisibleUnit.SOFTMAX) {
            INDArray v1Sample = Activations.softMaxRows().apply(v1Mean);
            return new Pair<>(v1Mean,v1Sample);
        }

        else if(conf.getVisibleUnit() == VisibleUnit.BINARY) {
            INDArray v1Sample = Sampling.binomial(v1Mean, 1, conf.getRng());
            return new Pair<>(v1Mean,v1Sample);
        }


        throw new IllegalStateException("Visible type must either be binary,gaussian, softmax, or linear");

    }

    /**
     * Calculates the activation of the visible :
     * sigmoid(v * W + hbias)
     * @param v the visible layer
     * @return the approximated activations of the visible layer
     */
    public INDArray propUp(INDArray v) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray hBias = getParam(PretrainParamInitializer.BIAS_KEY);

        if(conf.getVisibleUnit() == VisibleUnit.GAUSSIAN)
            this.sigma = v.var(0).divi(input.rows());

        INDArray preSig = v.mmul(W);
        if(conf.isConcatBiases())
            preSig = Nd4j.hstack(preSig,hBias);
        else
            preSig.addiRowVector(hBias);


        if(conf.getHiddenUnit() == HiddenUnit.RECTIFIED) {
            preSig = Transforms.max(preSig);
            return preSig;
        }

        else if(conf.getHiddenUnit() == HiddenUnit.GAUSSIAN) {
            INDArray add =  preSig.add(Nd4j.randn(preSig.rows(), preSig.columns(),conf.getRng()));
            preSig.addi(add);
            return preSig;
        }

        else if(conf.getHiddenUnit() == HiddenUnit.BINARY) {
            return sigmoid(preSig);
        }

        else if(conf.getHiddenUnit() == HiddenUnit.SOFTMAX)
            return Activations.softMaxRows().apply(preSig);

        throw new IllegalStateException("Hidden unit type should either be binary, gaussian, or rectified linear");

    }




    /**
     * Calculates the activation of the hidden:
     * sigmoid(h * W + vbias)
     * @param h the hidden layer
     * @return the approximated output of the hidden layer
     */
    public INDArray propDown(INDArray h) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray vBias = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);

        INDArray vMean = h.mmul(W.transpose());
        if(conf.isConcatBiases())
            vMean = Nd4j.hstack(vMean, vBias);
        else
            vMean.addiRowVector(vBias);

        if(conf.getVisibleUnit()  == VisibleUnit.GAUSSIAN) {
            INDArray sample = Sampling.normal(conf.getRng(), vMean, 1.0);
            vMean.addi(sample);
            return vMean;
        }
        else if(conf.getVisibleUnit() == VisibleUnit.LINEAR) {
            vMean = Sampling.normal(conf.getRng(),vMean,1);
            return vMean;
        }

        else if(conf.getVisibleUnit() == VisibleUnit.BINARY) {
            return sigmoid(vMean);
        }

        else if(conf.getVisibleUnit() == VisibleUnit.SOFTMAX) {
            return Activations.softMaxRows().apply(vMean);
        }


        throw new IllegalStateException("Visible unit type should either be binary or gaussian");

    }



    /**
     * Reconstructs the visible input.
     * A reconstruction is a propdown of the reconstructed hidden input.
     * @param v the visible input
     * @return the reconstruction of the visible input
     */
    @Override
    public INDArray transform(INDArray v) {
        //reconstructed: propUp ----> hidden propDown to transform
        return propDown(propUp(v));
    }



    /**
     * Note: k is the first input iken params.
     */
    @Override
    public void fit(INDArray input) {
        if(input != null)
            this.input = Transforms.stabilize(input, 1);

        if(conf.getVisibleUnit() == VisibleUnit.GAUSSIAN) {
            this.sigma = input.var(0);
            this.sigma.divi(input.rows());
        }

        super.fit(input);
    }


    @Override
    public String toString() {
        return "RBM{" +
                ", visibleType=" + conf.getVisibleUnit() +
                ", hiddenType=" + conf.getVisibleUnit() +
                ", sigma=" + sigma +
                ", hiddenSigma=" + hiddenSigma +
                "} " + super.toString();
    }



    @Override
    public void iterate(INDArray input) {
        if(conf.getVisibleUnit() == VisibleUnit.GAUSSIAN)
            this.sigma = input.var(0).divi(input.rows());

        this.input = input;
        contrastiveDivergence();
    }







}
