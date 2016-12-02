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
 * Created by Alex on 25/11/2016.
 */
@Data
public class VariationalAutoencoder extends BasePretrainNetwork {

    private int[] encoderLayerSizes;
    private int[] decoderLayerSizes;
    private ReconstructionDistribution outputDistribution;

    private VariationalAutoencoder(Builder builder){
        super(builder);
        this.encoderLayerSizes = builder.encoderLayerSizes;
        this.decoderLayerSizes = builder.decoderLayerSizes;
        this.outputDistribution = builder.outputDistribution;
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

        public Builder encoderLayerSizes(int... encoderLayerSizes){
            this.encoderLayerSizes = encoderLayerSizes;
            return this;
        }

        public Builder decoderLayerSizes(int... decoderLayerSizes){
            this.decoderLayerSizes = decoderLayerSizes;
            return this;
        }

        public Builder reconstructionDistribution(ReconstructionDistribution distribution){
            this.outputDistribution = distribution;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public VariationalAutoencoder build() {
            return new VariationalAutoencoder(this);
        }
    }
}
