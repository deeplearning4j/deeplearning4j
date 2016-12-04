package org.deeplearning4j.nn.conf.layers.variational;

import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.LayerValidation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Variational Autoencoder layer
 *<p>
 * See: Kingma & Welling, 2013: Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114
 *<p>
 * This implementation allows multiple encoder and decoder layers, the number and sizes of which can be set independently.
 *
 * @author Alex Black
 */
@Data
public class VariationalAutoencoder extends BasePretrainNetwork {

    private int[] encoderLayerSizes;
    private int[] decoderLayerSizes;
    private ReconstructionDistribution outputDistribution;
    private String pzxActivationFunction;

    private VariationalAutoencoder(Builder builder){
        super(builder);
        this.encoderLayerSizes = builder.encoderLayerSizes;
        this.decoderLayerSizes = builder.decoderLayerSizes;
        this.outputDistribution = builder.outputDistribution;
        this.pzxActivationFunction = builder.pzxActivationFunction;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("VariationalAutoencoder", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder ret = new org.deeplearning4j.nn.layers.variational.VariationalAutoencoder(conf);

        ret.setListeners(iterationListeners);
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
    public double getLearningRateByParam(String paramName) {
        if(paramName.endsWith("b")){
            if(!Double.isNaN(biasLearningRate)){
                //Bias learning rate has been explicitly set
                return biasLearningRate;
            } else {
                return learningRate;
            }
        } else {
            return learningRate;
        }
    }

    @Override
    public double getL1ByParam(String paramName) {
        if(paramName.endsWith(VariationalAutoencoderParamInitializer.BIAS_KEY_SUFFIX)) return 0.0;
        return l1;
    }

    @Override
    public double getL2ByParam(String paramName) {
        if(paramName.endsWith(VariationalAutoencoderParamInitializer.BIAS_KEY_SUFFIX)) return 0.0;
        return l2;
    }

    public static class Builder extends BasePretrainNetwork.Builder<Builder>{

        private int[] encoderLayerSizes = new int[]{100};
        private int[] decoderLayerSizes = new int[]{100};
        private ReconstructionDistribution outputDistribution = new GaussianReconstructionDistribution("tanh");
        private String pzxActivationFunction = "identity";

        /**
         * Size of the encoder layers, in units. Each encoder layer is functionally equivalent to a {@link org.deeplearning4j.nn.conf.layers.DenseLayer}.
         * Typically the number and size of the decoder layers (set via {@link #decoderLayerSizes(int...)} is similar to the encoder layers.
         *
         * @param encoderLayerSizes    Size of each encoder layer in the variational autoencoder
         */
        public Builder encoderLayerSizes(int... encoderLayerSizes){
            if(encoderLayerSizes == null || encoderLayerSizes.length < 1){
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
        public Builder decoderLayerSizes(int... decoderLayerSizes){
            if(encoderLayerSizes == null || encoderLayerSizes.length < 1){
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
        public Builder reconstructionDistribution(ReconstructionDistribution distribution){
            this.outputDistribution = distribution;
            return this;
        }

        /**
         * Activation function for the input to P(z|data).<br>
         * Care should be taken with this, as some activation functions (relu, etc) are not suitable due to being
         * bounded in range [0,infinity).
         *
         * @param activationFunction    Activation function for p(z|x)
         * @return
         */
        public Builder pzxActivationFunction(String activationFunction){
            this.pzxActivationFunction = activationFunction;
            return this;
        }

        /**
         * Set the size of the VAE state Z. This is the output size during standard forward pass, and the size of the
         * distribution P(Z|data) during pretraining.
         *
         * @param nOut    Size of P(Z|data) and output size
         */
        @Override
        public Builder nOut(int nOut){
            super.nOut(nOut);
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public VariationalAutoencoder build() {
            return new VariationalAutoencoder(this);
        }
    }
}
