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

package org.deeplearning4j.nn.conf.layers.variational;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Collection;
import java.util.Map;

/**
 * Variational Autoencoder layer
 *<p>
 * See: Kingma & Welling, 2013: Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114
 *<p>
 * This implementation allows multiple encoder and decoder layers, the number and sizes of which can be set independently.
 * <p>
 * A note on scores during pretraining: This implementation minimizes the negative of the variational lower bound objective
 * as described in Kingma & Welling; the mathematics in that paper is based on maximization of the variational lower bound instead.
 * Thus, scores reported during pretraining in DL4J are the negative of the variational lower bound equation in the paper.
 * The backpropagation and learning procedure is otherwise as described there.
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class VariationalAutoencoder extends BasePretrainNetwork {

    private int[] encoderLayerSizes;
    private int[] decoderLayerSizes;
    private ReconstructionDistribution outputDistribution;
    private IActivation pzxActivationFn;
    private int numSamples;

    private VariationalAutoencoder(Builder builder) {
        super(builder);
        this.encoderLayerSizes = builder.encoderLayerSizes;
        this.decoderLayerSizes = builder.decoderLayerSizes;
        this.outputDistribution = builder.outputDistribution;
        this.pzxActivationFn = builder.pzxActivationFn;
        this.numSamples = builder.numSamples;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("VariationalAutoencoder", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder ret =
                        new org.deeplearning4j.nn.layers.variational.VariationalAutoencoder(conf);

        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return VariationalAutoencoderParamInitializer.getInstance();
    }

    @Override
    public double getL1ByParam(String paramName) {
        if (paramName.endsWith(VariationalAutoencoderParamInitializer.BIAS_KEY_SUFFIX))
            return l1Bias;
        return l1;
    }

    @Override
    public double getL2ByParam(String paramName) {
        if (paramName.endsWith(VariationalAutoencoderParamInitializer.BIAS_KEY_SUFFIX))
            return l2Bias;
        return l2;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        if (paramName.startsWith(VariationalAutoencoderParamInitializer.DECODER_PREFIX)) {
            return true;
        }
        if (paramName.startsWith(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_PREFIX)) {
            return true;
        }
        if (paramName.startsWith(VariationalAutoencoderParamInitializer.PXZ_PREFIX)) {
            return true;
        }
        return false;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //For training: we'll assume unsupervised pretraining, as this has higher memory requirements

        InputType outputType = getOutputType(-1, inputType);

        val actElementsPerEx = outputType.arrayElementsPerExample();
        val numParams = initializer().numParams(this);
        int updaterStateSize = (int) getIUpdater().stateSize(numParams);

        int inferenceWorkingMemSizePerEx = 0;
        //Forward pass size through the encoder:
        for (int i = 1; i < encoderLayerSizes.length; i++) {
            inferenceWorkingMemSizePerEx += encoderLayerSizes[i];
        }

        //Forward pass size through the decoder, during training
        //p(Z|X) mean and stdev; pzxSigmaSquared, pzxSigma -> all size equal to nOut
        long decoderFwdSizeWorking = 4 * nOut;
        //plus, nSamples * decoder size
        //For each decoding: random sample (nOut), z (nOut), activations for each decoder layer
        decoderFwdSizeWorking += numSamples * (2 * nOut + ArrayUtil.sum(getDecoderLayerSizes()));
        //Plus, component of score
        decoderFwdSizeWorking += nOut;



        //Backprop size through the decoder and decoder: approx. 2x forward pass size
        long trainWorkingMemSize = 2 * (inferenceWorkingMemSizePerEx + decoderFwdSizeWorking);


        if (getIDropout() != null) {
            if (false) {
                //TODO drop connect
                //Dup the weights... note that this does NOT depend on the minibatch size...
            } else {
                //Assume we dup the input
                trainWorkingMemSize += inputType.arrayElementsPerExample();
            }
        }

        return new LayerMemoryReport.Builder(layerName, VariationalAutoencoder.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize)
                        .workingMemory(0, inferenceWorkingMemSizePerEx, 0, trainWorkingMemSize)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    public static class Builder extends BasePretrainNetwork.Builder<Builder> {

        private int[] encoderLayerSizes = new int[] {100};
        private int[] decoderLayerSizes = new int[] {100};
        private ReconstructionDistribution outputDistribution = new GaussianReconstructionDistribution(Activation.TANH);
        private IActivation pzxActivationFn = new ActivationIdentity();
        private int numSamples = 1;

        /**
         * Size of the encoder layers, in units. Each encoder layer is functionally equivalent to a {@link org.deeplearning4j.nn.conf.layers.DenseLayer}.
         * Typically the number and size of the decoder layers (set via {@link #decoderLayerSizes(int...)} is similar to the encoder layers.
         *
         * @param encoderLayerSizes    Size of each encoder layer in the variational autoencoder
         */
        public Builder encoderLayerSizes(int... encoderLayerSizes) {
            if (encoderLayerSizes == null || encoderLayerSizes.length < 1) {
                throw new IllegalArgumentException("Encoder layer sizes array must have length > 0");
            }
            this.encoderLayerSizes = encoderLayerSizes;
            return this;
        }

        /**
         * Size of the decoder layers, in units. Each decoder layer is functionally equivalent to a {@link org.deeplearning4j.nn.conf.layers.DenseLayer}.
         * Typically the number and size of the decoder layers is similar to the encoder layers (set via {@link #encoderLayerSizes(int...)}.
         *
         * @param decoderLayerSizes    Size of each deccoder layer in the variational autoencoder
         */
        public Builder decoderLayerSizes(int... decoderLayerSizes) {
            if (decoderLayerSizes == null || decoderLayerSizes.length < 1) {
                throw new IllegalArgumentException("Decoder layer sizes array must have length > 0");
            }
            this.decoderLayerSizes = decoderLayerSizes;
            return this;
        }

        /**
         * The reconstruction distribution for the data given the hidden state - i.e., P(data|Z).<br>
         * This should be selected carefully based on the type of data being modelled. For example:<br>
         * - {@link GaussianReconstructionDistribution} + {identity or tanh} for real-valued (Gaussian) data<br>
         * - {@link BernoulliReconstructionDistribution} + sigmoid for binary-valued (0 or 1) data<br>
         *
         * @param distribution    Reconstruction distribution
         */
        public Builder reconstructionDistribution(ReconstructionDistribution distribution) {
            this.outputDistribution = distribution;
            return this;
        }

        /**
         * Configure the VAE to use the specified loss function for the reconstruction, instead of a ReconstructionDistribution.
         * Note that this is NOT following the standard VAE design (as per Kingma & Welling), which assumes a probabilistic
         * output - i.e., some p(x|z). It is however a valid network configuration, allowing for optimization of more traditional
         * objectives such as mean squared error.<br>
         * Note: clearly, setting the loss function here will override any previously set recontruction distribution
         *
         * @param outputActivationFn Activation function for the output/reconstruction
         * @param lossFunction       Loss function to use
         */
        public Builder lossFunction(IActivation outputActivationFn, LossFunctions.LossFunction lossFunction) {
            return lossFunction(outputActivationFn, lossFunction.getILossFunction());
        }

        /**
         * Configure the VAE to use the specified loss function for the reconstruction, instead of a ReconstructionDistribution.
         * Note that this is NOT following the standard VAE design (as per Kingma & Welling), which assumes a probabilistic
         * output - i.e., some p(x|z). It is however a valid network configuration, allowing for optimization of more traditional
         * objectives such as mean squared error.<br>
         * Note: clearly, setting the loss function here will override any previously set recontruction distribution
         *
         * @param outputActivationFn Activation function for the output/reconstruction
         * @param lossFunction       Loss function to use
         */
        public Builder lossFunction(Activation outputActivationFn, LossFunctions.LossFunction lossFunction) {
            return lossFunction(outputActivationFn.getActivationFunction(), lossFunction.getILossFunction());
        }

        /**
         * Configure the VAE to use the specified loss function for the reconstruction, instead of a ReconstructionDistribution.
         * Note that this is NOT following the standard VAE design (as per Kingma & Welling), which assumes a probabilistic
         * output - i.e., some p(x|z). It is however a valid network configuration, allowing for optimization of more traditional
         * objectives such as mean squared error.<br>
         * Note: clearly, setting the loss function here will override any previously set recontruction distribution
         *
         * @param outputActivationFn Activation function for the output/reconstruction
         * @param lossFunction       Loss function to use
         */
        public Builder lossFunction(IActivation outputActivationFn, ILossFunction lossFunction) {
            return reconstructionDistribution(new LossFunctionWrapper(outputActivationFn, lossFunction));
        }

        /**
         * Activation function for the input to P(z|data).<br>
         * Care should be taken with this, as some activation functions (relu, etc) are not suitable due to being
         * bounded in range [0,infinity).
         *
         * @param activationFunction    Activation function for p(z|x)
         */
        public Builder pzxActivationFn(IActivation activationFunction) {
            this.pzxActivationFn = activationFunction;
            return this;
        }

        /**
         * Activation function for the input to P(z|data).<br>
         * Care should be taken with this, as some activation functions (relu, etc) are not suitable due to being
         * bounded in range [0,infinity).
         *
         * @param activation    Activation function for p(z|x)
         */
        public Builder pzxActivationFunction(Activation activation) {
            return pzxActivationFn(activation.getActivationFunction());
        }

        /**
         * Set the size of the VAE state Z. This is the output size during standard forward pass, and the size of the
         * distribution P(Z|data) during pretraining.
         *
         * @param nOut    Size of P(Z|data) and output size
         */
        @Override
        public Builder nOut(int nOut) {
            super.nOut(nOut);
            return this;
        }

        /**
         * Set the number of samples per data point (from VAE state Z) used when doing pretraining. Default value: 1.
         * <p>
         * This is parameter L from Kingma and Welling: "In our experiments we found that the number of samples L per
         * datapoint can be set to 1 as long as the minibatch size M was large enough, e.g. M = 100."
         *
         * @param numSamples    Number of samples per data point for pretraining
         */
        public Builder numSamples(int numSamples) {
            this.numSamples = numSamples;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public VariationalAutoencoder build() {
            return new VariationalAutoencoder(this);
        }
    }
}
