/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers.feedforward.rbm;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.Dropout;
import org.deeplearning4j.util.RBMUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;


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
public  class RBM extends BasePretrainNetwork<org.deeplearning4j.nn.conf.layers.RBM> {

    private long seed;

    public RBM(NeuralNetConfiguration conf) {
        super(conf);
        this.seed = conf.getSeed();
    }

    public RBM(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        this.seed = conf.getSeed();
    }

    /**
     *
     */
    //variance matrices for gaussian visible/hidden units
    @Deprecated
    protected INDArray sigma, hiddenSigma;


    /**
     * Contrastive divergence revolves around the idea
     * of approximating the log likelihood around x1(input) with repeated sampling.
     * Given is an energy based model: the higher k is (the more we sample the model)
     * the more we lower the energy (increase the likelihood of the model)
     * <p>
     * and lower the likelihood (increase the energy) of the hidden samples.
     * <p>
     * Other insights:
     * CD - k involves keeping the first k samples of a gibbs sampling of the model.
     */
    @Deprecated
    public void contrastiveDivergence() {
        Gradient gradient = gradient();
        getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY).subi(gradient.gradientForVariable().get(PretrainParamInitializer.VISIBLE_BIAS_KEY));
        getParam(PretrainParamInitializer.BIAS_KEY).subi(gradient.gradientForVariable().get(PretrainParamInitializer.BIAS_KEY));
        getParam(PretrainParamInitializer.WEIGHT_KEY).subi(gradient.gradientForVariable().get(PretrainParamInitializer.WEIGHT_KEY));
    }


    @Override
    public void computeGradientAndScore() {
        int k = layerConf().getK();

        //POSITIVE PHASE
        // hprob0, hstate0
        Pair<INDArray, INDArray> probHidden = sampleHiddenGivenVisible(input());

		/*
		 * Start the gibbs sampling.
		 */
//        INDArray chainStart = probHidden.getSecond();
        INDArray chainStart = probHidden.getFirst();

		/*
		 * Note that at a later date, we can explore alternative methods of
		 * storing the chain transitions for different kinds of sampling
		 * and exploring the search space.
		 */
        Pair<Pair<INDArray, INDArray>, Pair<INDArray, INDArray>> matrices;
        //negative value samples
        INDArray negVProb = null;
        //negative value samples
        INDArray negVSamples = null;
        //negative hidden means or expected values
        INDArray negHProb = null;
        //negative hidden samples
        INDArray negHSamples = null;

		/*
		 * K steps of gibbs sampling. This is the positive phase of contrastive divergence.
		 *
		 * There are 4 matrices being computed for each gibbs sampling.
		 * The samples from both the positive and negative phases and their expected values
		 * or averages.
		 *
		 */

        for (int i = 0; i < k; i++) {

            //NEGATIVE PHASE
            if (i == 0)
                matrices = gibbhVh(chainStart);
            else
                matrices = gibbhVh(negHSamples);

            //get the cost updates for sampling in the chain after k iterations
            negVProb = matrices.getFirst().getFirst();
            negVSamples = matrices.getFirst().getSecond();
            negHProb = matrices.getSecond().getFirst();
            negHSamples = matrices.getSecond().getSecond();
        }

		/*
		 * Update gradient parameters - note taking mean based on batchsize is handled in LayerUpdater
		 */
        INDArray wGradient = input().transposei().mmul(probHidden.getFirst()).subi(
                negVProb.transpose().mmul(negHProb)
        );

        INDArray hBiasGradient;

        if (layerConf().getSparsity() != 0)
            //all hidden units must stay around this number
            hBiasGradient = probHidden.getFirst().rsub(layerConf().getSparsity()).sum(0);
        else
            //update rule: the expected values of the hidden input - the negative hidden  means adjusted by the learning rate
            hBiasGradient = probHidden.getFirst().sub(negHProb).sum(0);

        //update rule: the expected values of the input - the negative samples adjusted by the learning rate
        INDArray delta = input.sub(negVProb);
        INDArray vBiasGradient = delta.sum(0);

        if (conf.isPretrain()) {
            wGradient.negi();
            hBiasGradient.negi();
            vBiasGradient.negi();
        }

        gradient = createGradient(wGradient, vBiasGradient, hBiasGradient);

        setScoreWithZ(negVSamples); // this is compared to input on

        if(trainingListeners != null && trainingListeners.size() > 0){
            for(TrainingListener tl : trainingListeners){
                tl.onBackwardPass(this);
            }
        }
    }

    /**
     * Gibbs sampling step: hidden ---> visible ---> hidden
     *
     * @param h the hidden input
     * @return the expected values and samples of both the visible samples given the hidden
     * and the new hidden input and expected values
     */
    public Pair<Pair<INDArray, INDArray>, Pair<INDArray, INDArray>> gibbhVh(INDArray h) {
        Pair<INDArray, INDArray> v1MeanAndSample = sampleVisibleGivenHidden(h);
        INDArray negVProb = v1MeanAndSample.getFirst();

        Pair<INDArray, INDArray> h1MeanAndSample = sampleHiddenGivenVisible(negVProb);
        return new Pair<>(v1MeanAndSample, h1MeanAndSample);
    }

    /**
     * Binomial sampling of the hidden values given visible
     *
     * @param v the visible values
     * @return a binomial distribution containing the expected values and the samples
     */
    @Override
    public Pair<INDArray, INDArray> sampleHiddenGivenVisible(INDArray v) {
        INDArray hProb = propUp(v);
        INDArray hSample;
        Distribution dist;

        switch (layerConf().getHiddenUnit()) {
            case IDENTITY: {
                hSample = hProb;
                break;
            }
            case BINARY: {
                dist = Nd4j.getDistributions().createBinomial(1, hProb);
                dist.reseedRandomGenerator(seed);
                hSample = dist.sample(hProb.shape());
                break;
            }
            case GAUSSIAN: {
                dist = Nd4j.getDistributions().createNormal(hProb, 1);
                dist.reseedRandomGenerator(seed);
                hSample = dist.sample(hProb.shape());
                break;
            }
            case RECTIFIED: {
                INDArray sigH1Mean = sigmoid(hProb);
		/*
		 * Rectified linear part
		 */
                INDArray sqrtSigH1Mean = sqrt(sigH1Mean);
                INDArray sample = Nd4j.getDistributions().createNormal(hProb, 1).sample(hProb.shape());
                sample.muli(sqrtSigH1Mean);
                hSample = hProb.add(sample);
                hSample = max(hSample, 0.0);
                break;
            }
            case SOFTMAX: {
                hSample = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", hProb));
                break;
            }
            default:
                throw new IllegalStateException("Hidden unit type must either be Binary, Gaussian, SoftMax or Rectified");
        }

        return new Pair<>(hProb, hSample);
    }

    /**
     * Guess the visible values given the hidden
     *
     * @param h the hidden units
     * @return a visible mean and sample relative to the hidden states
     * passed in
     */
    @Override
    public Pair<INDArray, INDArray> sampleVisibleGivenHidden(INDArray h) {
        INDArray vProb = propDown(h);
        INDArray vSample;

        switch (layerConf().getVisibleUnit()) {
            case IDENTITY: {
                vSample = vProb;
                break;
            }
            case BINARY: {
                Distribution dist = Nd4j.getDistributions().createBinomial(1, vProb);
                dist.reseedRandomGenerator(seed);
                vSample = dist.sample(vProb.shape());
                break;
            }
            case GAUSSIAN:
            case LINEAR: {
                Distribution dist = Nd4j.getDistributions().createNormal(vProb, 1);
                dist.reseedRandomGenerator(seed);
                vSample = dist.sample(vProb.shape());
                // this also works but needs reseedRnadomGenerator applied before sampling: Nd4j.getDistributions().createNormal(v1Mean, 1).sample(v1Mean.shape());
                break;
            }
            case SOFTMAX: {
                vSample = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", vProb));
                break;
            }
            default: {
                throw new IllegalStateException("Visible type must be one of Binary, Gaussian, SoftMax or Linear");
            }
        }

        return new Pair<>(vProb, vSample);

    }

    public INDArray preOutput(INDArray v, boolean training) {
        INDArray hBias = getParam(PretrainParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        if(training && conf.isUseDropConnect() && conf.getLayer().getDropOut() > 0) {
            W = Dropout.applyDropConnect(this,DefaultParamInitializer.WEIGHT_KEY);
        }
        return v.mmul(W).addiRowVector(hBias);
    }

    /**
     * Calculates the activation of the visible :
     * sigmoid(v * W + hbias)
     * @param v the visible layer
     * @return the approximated activations of the visible layer
     */
    public INDArray propUp(INDArray v) {
        return propUp(v,true);
    }

    /**
     * Calculates the activation of the visible :
     * sigmoid(v * W + hbias)
     * @param v the visible layer
     * @return the approximated activations of the visible layer
     */
    public INDArray propUp(INDArray v, boolean training) {
        INDArray preSig = preOutput(v, training);

        switch (layerConf().getHiddenUnit()) {
            case IDENTITY:
                return preSig;
            case BINARY:
                return sigmoid(preSig);
            case GAUSSIAN:
                Distribution dist = Nd4j.getDistributions().createNormal(preSig, 1);
                dist.reseedRandomGenerator(seed);
                preSig = dist.sample(preSig.shape());
                return preSig;
            case RECTIFIED:
                preSig = max(preSig, 0.0);
                return preSig;
            case SOFTMAX:
                return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", preSig));
            default:
                throw new IllegalStateException("Hidden unit type should either be binary, gaussian, or rectified linear");
        }

    }

    public INDArray propUpDerivative(INDArray z) {
        switch (layerConf().getHiddenUnit()) {
            case IDENTITY:
                return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("identity", z).derivative());
            case BINARY:
                return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", z).derivative());
            case GAUSSIAN: {
                Distribution dist = Nd4j.getDistributions().createNormal(z, 1);
                dist.reseedRandomGenerator(seed);
                INDArray gaussian = dist.sample(z.shape());
                INDArray derivative = z.mul(-2).mul(gaussian);
                return derivative;
            }
            case RECTIFIED:
                return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("relu", z).derivative());
            case SOFTMAX:
                return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", z).derivative());
            default:
                throw new IllegalStateException("Hidden unit type should either be binary, gaussian, or rectified linear");
        }

    }

    /**
     * Calculates the activation of the hidden:
     * activation(h * W + vbias)
     * @param h the hidden layer
     * @return the approximated output of the hidden layer
     */
    public INDArray propDown(INDArray h) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY).transpose();
        INDArray vBias = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);

        INDArray vMean = h.mmul(W).addiRowVector(vBias);

        switch (layerConf().getVisibleUnit()) {
            case IDENTITY:
                return vMean;
            case BINARY:
                return sigmoid(vMean);
            case GAUSSIAN:
                Distribution dist = Nd4j.getDistributions().createNormal(vMean, 1);
                dist.reseedRandomGenerator(seed);
                vMean = dist.sample(vMean.shape());
                return vMean;
            case LINEAR:
                return vMean;
            case SOFTMAX:
                return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", vMean));
            default:
                throw new IllegalStateException("Visible unit type should either be binary or gaussian");
        }

    }

    /**
     * Reconstructs the visible INPUT.
     * A reconstruction is a propdown of the reconstructed hidden input.
     * @param  training true or false
     * @return the reconstruction of the visible input
     */
    @Override
    public INDArray activate(boolean training) {
        if(training && conf.getLayer().getDropOut() > 0.0) {
            Dropout.applyDropout(input,conf.getLayer().getDropOut());
        }
        //reconstructed: propUp ----> hidden propDown to transform
        INDArray propUp = propUp(input, training);
        return propUp;
    }

    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {
        //If this layer is layer L, then epsilon is (w^(L+1)*(d^(L+1))^T) (or equivalent)
        INDArray z = preOutput(input, true);
        INDArray activationDerivative = propUpDerivative(z);
        INDArray delta = epsilon.muli(activationDerivative);

        if(maskArray != null){
            delta.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();

        INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);    //f order
        Nd4j.gemm(input,delta,weightGrad,true,false,1.0,0.0);
        INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
        biasGrad.assign(delta.sum(0));
        INDArray vBiasGradient = gradientViews.get(PretrainParamInitializer.VISIBLE_BIAS_KEY);

        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);
        ret.gradientForVariable().put(PretrainParamInitializer.VISIBLE_BIAS_KEY, vBiasGradient);

        INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(delta.transpose()).transpose();

        return new Pair<>(ret,epsilonNext);
    }


    @Deprecated
    @Override
    public void iterate(INDArray input) {
        if(layerConf().getVisibleUnit() == org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN)
            this.sigma = input.var(0).divi(input.rows());

        this.input = input.dup();
        applyDropOutIfNecessary(true);
        contrastiveDivergence();
    }

    @Deprecated
    @Override
    public Layer transpose() {
        RBM r = (RBM) super.transpose();
        org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit h = RBMUtil.inverse(layerConf().getVisibleUnit());
        org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit v = RBMUtil.inverse(layerConf().getHiddenUnit());
        if(h == null)
            h = layerConf().getHiddenUnit();
        if(v == null)
            v = layerConf().getVisibleUnit();

        r.layerConf().setHiddenUnit(h);
        r.layerConf().setVisibleUnit(v);

        //biases:
        INDArray vb = getParam(DefaultParamInitializer.BIAS_KEY).dup();
        INDArray b = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY).dup();
        r.setParam(PretrainParamInitializer.VISIBLE_BIAS_KEY,vb);
        r.setParam(DefaultParamInitializer.BIAS_KEY,b);

        r.sigma = sigma;
        r.hiddenSigma = hiddenSigma;
        return r;
    }

    @Override
    public boolean isPretrainLayer() {
        return true;
    }


}
