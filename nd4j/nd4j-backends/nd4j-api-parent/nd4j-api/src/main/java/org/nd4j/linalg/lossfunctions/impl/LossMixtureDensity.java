/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.lossfunctions.impl;

import lombok.Data;
import lombok.EqualsAndHashCode;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.strict.OldSoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 * This is a cost function associated with a mixture-density network.
 * For background, this is inspired by Bishop's work pioneering the mixture
 * density network.  The essence of the idea is that the cost function attempts
 * to model the output as if it were a mixture of gaussian probability
 * densities.  The network attempts to converge on a cost function which
 * involves the negative log likelihood of the output fitting a set of data
 * and estimates the "alpha" contribution to of each of the distributions
 * the 'mu' (mean) and 'sigma' (standard deviation) of each of the
 * distributions.
 * 
 * For a full description of the technique, refer to Bishop's work.
 * 
 * Bishop CM. Mixture density networks,
 * Neural Computing Research Group Report:
 * NCRG/94/004, Aston University, Birmingham, 1994
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf
 * 
 * There is no public constructor, please use the builder to create an
 * approriate mixture loss function for the number of outputs and number
 * of mixtures you would like to fit.
 * 
 * Note that this means that the output
 * layer must provide (labelWidth+2)*mixtures output values in order to describe
 * the distributions of vectors of width labelWidth.
 * Please ensure that the size of the output layer matches the number of
 * mixtures provided.
 * 
 * @author Jonathan Arney
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
public class LossMixtureDensity extends DifferentialFunction implements ILossFunction {

    private  int mMixtures;
    private  int mLabelWidth;
    private static final double SQRT_TWO_PI = Math.sqrt(2 * Math.PI);

    public LossMixtureDensity() {
    }

    /**
     * This method constructs a mixture density cost function
     * which causes the network to learn a mixture of gaussian distributions
     * for each network output.  The network will learn the 'alpha' (weight
     * for each distribution), the 'mu' or 'mean' of each distribution,
     * and the 'sigma' (standard-deviation) of the mixture.  Together,
     * this distribution can be sampled according to the probability density
     * learned by the network.
     * 
     * @param mixtures Number of gaussian mixtures to model.
     * @param labelWidth Size of the labels vector for each sample.
     */
    private LossMixtureDensity(@JsonProperty("mixtures") int mixtures, @JsonProperty("labelWidth") int labelWidth) {
        mMixtures = mixtures;
        mLabelWidth = labelWidth;
    }

    /**
     * This class is a data holder for the mixture density
     * components for convenient manipulation.
     * These are organized as rank-3 matrices with shape
     * [nSamples, nLabelsPerSample, nMixturesPerLabel]
     * and refer to the 'alpha' (weight of that gaussian), 'mu' (mean for that
     * gaussian), and 'sigma' (standard-deviation for that gaussian).
     */
    @Data
    public static class MixtureDensityComponents {
        private INDArray alpha;
        private INDArray mu;
        private INDArray sigma;
    }

    // This method extracts the "alpha", "mu", and "sigma" from the
    // output of the neural network.
    // This is done manually, but it should ultimately be done
    // through Nd4j operations in order to increase performance.
    public MixtureDensityComponents extractComponents(INDArray output) {
        long outputSize = output.size(1);
        if (outputSize != (mLabelWidth + 2) * mMixtures) {
            throw new IllegalArgumentException(
                            "Network output size " + outputSize + " must be (labels+2)*mixtures where labels = "
                                            + mLabelWidth + " and mixtures = " + mMixtures);
        }

        MixtureDensityComponents mdc = new MixtureDensityComponents();

        // Output is 2 dimensional (samples, labels)
        //
        // For each label vector of length 'labels', we will have
        // an output vector of length '(labels + 2) * nMixtures.
        // The first nMixtures outputs will correspond to the 'alpha' for each mixture.
        // The second nMixtures outputs will correspond to the 'sigma' and the last nMixtures*labels
        // will correspond to the 'mu' (mean) of the output.

        // Reorganize these.
        // alpha = samples, 0 to nMixtures
        // mu = samples, nMixtures to 2*nMixtures
        // sigma = samples, 2*nMixtures to (labels + 2)*nMixtures
        // Alpha is then sub-divided through reshape by mixtures per label and samples.

        mdc.alpha = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, mMixtures));
        mdc.sigma = output.get(NDArrayIndex.all(), NDArrayIndex.interval(mMixtures, 2 * mMixtures));
        mdc.mu = output.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * mMixtures, (mLabelWidth + 2) * mMixtures))
                        .reshape(output.size(0), mMixtures, mLabelWidth);

        // Alpha is a softmax because
        // the alpha should all sum to 1 for a given gaussian mixture.
        mdc.alpha = Nd4j.getExecutioner().execAndReturn(new OldSoftMax(mdc.alpha));

        // Mu comes directly from the network as an unmolested value.
        // Note that this effectively means that the output layer of
        // the network should have an activation function at least as large as
        // the expected values.  It is best for the output
        // layer to be an IDENTITY activation function.
        //mdc.mu = mdc.mu;

        // Sigma comes from the network as an exponential in order to
        // ensure that it is positive-definite.
        mdc.sigma = Transforms.exp(mdc.sigma);

        return mdc;
    }

    /**
     * Computes the aggregate score as a sum of all of the individual scores of
     * each of the labels against each of the outputs of the network.  For
     * the mixture density network, this is the negative log likelihood that
     * the given labels fall within the probability distribution described by
     * the mixture of gaussians of the network output.
     * @param labels Labels to score against the network.
     * @param preOutput Output of the network (before activation function has been called).
     * @param activationFn Activation function for the network.
     * @param mask Mask to be applied to labels (not used for MDN).
     * @param average Whether or not to return an average instead of a total score (not used).
     * @return Returns a single double which corresponds to the total score of all label values.
     */
    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {
        // The score overall consists of the
        // sum of the negative log likelihoods for each
        // of the individual labels.
        INDArray scoreArr = computeScoreArray(labels, preOutput, activationFn, mask);
        double score = scoreArr.sumNumber().doubleValue();
        if (average) {
            score /= scoreArr.size(0);
        }
        return score;
    }

    /**
     * This method returns the score for each of the given outputs against the
     * given set of labels.  For a mixture density network, this is done by
     * extracting the "alpha", "mu", and "sigma" components of each gaussian
     * and computing the negative log likelihood that the labels fall within
     * a linear combination of these gaussian distributions.  The smaller
     * the negative log likelihood, the higher the probability that the given
     * labels actually would fall within the distribution.  Therefore by
     * minimizing the negative log likelihood, we get to a position of highest
     * probability that the gaussian mixture explains the phenomenon.
     *
     * @param labels Labels give the sample output that the network should
     *               be trying to converge on.
     * @param preOutput The output of the last layer (before applying the activation function).
     * @param activationFn The activation function of the current layer.
     * @param mask Mask to apply to score evaluation (not supported for this cost function).
     * @return 
     */
    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray output = activationFn.getActivation(preOutput.dup(), false);
        MixtureDensityComponents mdc = extractComponents(output);
        INDArray scoreArr = negativeLogLikelihood(labels, mdc.alpha, mdc.mu, mdc.sigma);

        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask);
        }

        return scoreArr;
    }

    /**
     * This method returns the gradient of the cost function with respect to the
     * output from the previous layer.  For this cost function, the gradient
     * is derived from Bishop's paper "Mixture Density Networks" (1994) which
     * gives an elegant closed-form expression for the derivatives with respect
     * to each of the output components.
     * @param labels Labels to train on.
     * @param preOutput Output of neural network before applying the final activation function.
     * @param activationFn Activation function of output layer.
     * @param mask Mask to apply to gradients.
     * @return Gradient of cost function with respect to preOutput parameters.
     */
    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        long nSamples = labels.size(0);

        INDArray output = activationFn.getActivation(preOutput.dup(), false);

        MixtureDensityComponents mdc = extractComponents(output);

        INDArray gradient = Nd4j.zeros(nSamples, preOutput.columns());

        INDArray labelsMinusMu = labelsMinusMu(labels, mdc.mu);
        INDArray labelsMinusMuSquared = labelsMinusMu.mul(labelsMinusMu).sum(2);

        // This computes pi_i, see Bishop equation (30).
        // See http://www.plsyard.com/dealing-overflow-and-underflow-in-softmax-function/
        // this post for why we calculate the pi_i in this way.
        // With the exponential function here, we have to be very careful
        // about overflow/underflow considerations even with
        // fairly intermediate values.  Subtracting the max
        // here helps to ensure over/underflow does not happen here.
        // This isn't exactly a softmax because there's an 'alpha' coefficient
        // here, but the technique works, nonetheless.
        INDArray variance = mdc.sigma.mul(mdc.sigma);
        INDArray minustwovariance = variance.mul(2).negi();
        INDArray normalPart = mdc.alpha.div(Transforms.pow(mdc.sigma.mul(SQRT_TWO_PI), mLabelWidth));
        INDArray exponent = labelsMinusMuSquared.div(minustwovariance);
        INDArray exponentMax = exponent.max(1);
        exponent.subiColumnVector(exponentMax);
        INDArray pi = Transforms.exp(exponent).muli(normalPart);
        INDArray piDivisor = pi.sum(1);
        pi.diviColumnVector(piDivisor);

        // See Bishop equation (35)
        //INDArray dLdZAlpha = Nd4j.zeros(nSamples, nLabelsPerSample, mMixturesPerLabel); //mdc.alpha.sub(pi);
        INDArray dLdZAlpha = mdc.alpha.sub(pi);
        // See Bishop equation (38)
        INDArray dLdZSigma = (labelsMinusMuSquared.div(variance).subi(mLabelWidth)).muli(-1).muli(pi);
        // See Bishop equation (39)

        // This turned out to be way less efficient than
        // the simple 'for' loop here.
        //INDArray dLdZMu = pi
        //        .div(variance)
        //        .reshape(nSamples, mMixtures, 1)
        //        .repeat(2, mLabelWidth)
        //        .muli(labelsMinusMu)
        //        .negi()
        //        .reshape(nSamples, mMixtures * mLabelWidth);

        INDArray dLdZMu = Nd4j.create(nSamples, mMixtures, mLabelWidth);
        for (int k = 0; k < mLabelWidth; k++) {
            dLdZMu.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(k)},
                            labelsMinusMu.get(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(),
                                            NDArrayIndex.point(k)}).muli(pi).divi(variance).negi());
        }
        dLdZMu = dLdZMu.reshape(nSamples, mMixtures * mLabelWidth);

        // Place components of gradient into gradient holder.
        gradient.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(0, mMixtures)}, dLdZAlpha);
        gradient.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(mMixtures, mMixtures * 2)},
                        dLdZSigma);
        gradient.put(new INDArrayIndex[] {NDArrayIndex.all(),
                        NDArrayIndex.interval(mMixtures * 2, (mLabelWidth + 2) * mMixtures)}, dLdZMu);

        INDArray gradients = activationFn.backprop(preOutput, gradient).getFirst();

        if (mask != null) {
            LossUtil.applyMask(gradients, mask);
        }

        return gradients;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                    INDArray mask, boolean average) {
        double score = computeScore(labels, preOutput, activationFn, mask, average);
        INDArray gradient = computeGradient(labels, preOutput, activationFn, mask);
        Pair<Double, INDArray> returnCode = new Pair<>(score, gradient);
        return returnCode;
    }

    /**
     * The opName of this function
     *
     * @return
     */
    @Override
    public String name() {
        return "lossmixturedensity";
    }

    /**
     * This method returns an array consisting of each of the training samples,
     * for each label in each sample, the negative log likelihood of that
     * value falling within the given gaussian mixtures.
     * @param alpha
     * @param mu
     * @param sigma
     * @param labels
     * @return 
     */
    private INDArray negativeLogLikelihood(INDArray labels, INDArray alpha, INDArray mu, INDArray sigma) {
        INDArray labelsMinusMu = labelsMinusMu(labels, mu);
        INDArray diffsquared = labelsMinusMu.mul(labelsMinusMu).sum(2);
        INDArray phitimesalphasum = phi(diffsquared, sigma).muli(alpha).sum(1);

        // result = See Bishop(28,29)
        INDArray result = Transforms.log(phitimesalphasum).negi();
        return result;
    }

    private INDArray labelsMinusMu(INDArray labels, INDArray mu) {
        // Now that we have the mixtures, let's compute the negative
        // log likelihodd of the label against the 
        long nSamples = labels.size(0);
        long labelsPerSample = labels.size(1);

        // This worked, but was actually much
        // slower than the for loop below.
        // labels = samples, mixtures, labels
        // mu = samples, mixtures
        // INDArray labelMinusMu = labels
        //        .reshape('f', nSamples, labelsPerSample, 1)
        //        .repeat(2, mMixtures)
        //        .permute(0, 2, 1)
        //        .subi(mu);

        // The above code does the same thing as the loop below,
        // but it does it with index magix instead of a for loop.
        // It turned out to be way less efficient than the simple 'for' here.
        INDArray labelMinusMu = Nd4j.zeros(nSamples, mMixtures, labelsPerSample);
        for (int k = 0; k < mMixtures; k++) {
            labelMinusMu.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.point(k), NDArrayIndex.all()},
                            labels);
        }
        labelMinusMu.subi(mu);

        return labelMinusMu;
    }

    /**
     * This method calculates 'phi' which is the probability
     * density function (see Bishop 23)
     * @param diffSquared This is the 'x-mu' term of the Gaussian distribution (distance between 'x' and the mean value of the distribution).
     * @param sigma This is the standard deviation of the Gaussian distribution.
     * @return This returns an array of shape [nsamples, nlabels, ndistributions] which contains the probability density (phi) for each of the
     *         samples * labels * distributions for the given x, sigma, mu.
     */
    private INDArray phi(INDArray diffSquared, INDArray sigma) {
        // 1/sqrt(2PIs^2) * e^((in-u)^2/2*s^2)
        INDArray minustwovariance = sigma.mul(sigma).muli(2).negi();

        // This is phi_i(x,mu,sigma)
        INDArray likelihoods = Transforms.exp(diffSquared.divi(minustwovariance))
                        .divi(Transforms.pow(sigma.mul(SQRT_TWO_PI), (double) mLabelWidth));

        return likelihoods;
    }

    /**
     * Returns the number of gaussians this loss function
     * will attempt to find.
     * @return Number of gaussians to find.
     */
    @JsonProperty("mixtures")
    public int getNMixtures() {
        return mMixtures;
    }

    /**
     * Returns the width of each label vector.
     * @return Width of label vectors expected.
     */
    @JsonProperty("labelWidth")
    public int getLabelWidth() {
        return mLabelWidth;
    }

    @Override
    public String toString() {
        return "LossMixtureDensity(mixtures=" + mMixtures + ", labels=" + mLabelWidth + ")";
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int mGaussians = 0;
        private int mLabelWidth = 0;

        private Builder() {}

        /**
         * Specifies the number of gaussian functions to attempt
         * fitting against the data.
         * @param aGaussians Number of gaussian functions to fit.
         * @return DynamicCustomOpsBuilder.
         */
        public Builder gaussians(int aGaussians) {
            mGaussians = aGaussians;
            return this;
        }

        /**
         * Specifies the width of the labels vector which also corresponds
         * to the width of the 'mean' vector for each of the gaussian functions.
         * @param aLabelWidth Width of the labels vector.
         * @return DynamicCustomOpsBuilder.
         */
        public Builder labelWidth(int aLabelWidth) {
            mLabelWidth = aLabelWidth;
            return this;
        }

        /**
         * Creates a new instance of the mixture density
         * cost function.
         * @return A new mixture density cost function built with
         *         the specified parameters.
         */
        public LossMixtureDensity build() {
            if (mGaussians <= 0) {
                throw new IllegalArgumentException(
                                "Mixture density cost function must specify the number of mixtures to fit");
            }
            if (mLabelWidth <= 0) {
                throw new IllegalArgumentException(
                                "Mixture density cost function must specify the size of the labels vectors");
            }
            return new LossMixtureDensity(mGaussians, mLabelWidth);
        }
    }


    @Override
    public SDVariable[] outputVariables() {
        return new SDVariable[0];
    }

    @Override
    public SDVariable[] outputVariables(String baseName) {
        return new SDVariable[0];
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }



    @Override
    public String opName() {
        return name();
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op name found for " + opName());
    }
}
