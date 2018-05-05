package org.deeplearning4j.nn.conf.ocnn;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.layers.ocnn.OCNNParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collection;
import java.util.Map;

/**
 * An implementation of one class neural networks from:
 * https://arxiv.org/pdf/1802.06360.pdf
 *
 * The one class neural network approach is an extension of the standard output layer
 * with a single set of weights, an activation function, and a bias to:
 * 2 sets of weights, a learnable "r" parameter that is held static
 * 1 traditional set of weights.
 * 1 additional weight matrix
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties("lossFn")
public class OCNNOutputLayer extends BaseOutputLayer {
    //embedded hidden layer size
    //aka "K"
    private int hiddenSize;

    private double nu = 0.04;

    public OCNNOutputLayer(Builder builder) {
        super(builder);
        this.hiddenSize = builder.hiddenLayerSize;
        this.nu = builder.nu;

    }

    @JsonCreator
    public OCNNOutputLayer(@JsonProperty("hiddenSize") int hiddenSize,@JsonProperty("nu") double nu) {
        this.hiddenSize = hiddenSize;
        this.nu = nu;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("OCNNOutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        Layer ret = new org.deeplearning4j.nn.layers.ocnn.OCNNOutputLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public int getNOut() {
        //we don't change number of outputs here
        return 1;
    }

    @Override
    public ParamInitializer initializer() {
        return OCNNParamInitializer.getInstance();
    }


    @Override
    public double getL1ByParam(String paramName) {
        return l1;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return l2;
    }

    @NoArgsConstructor
    public static class Builder extends BaseOutputLayer.Builder<Builder> {
        protected  int hiddenLayerSize;
        protected  double nu = 0.04;

        public Builder nu(double nu) {
            this.nu = nu;
            return this;
        }

        public Builder hiddenLayerSize(int hiddenLayerSize) {
            this.hiddenLayerSize = hiddenLayerSize;
            return this;
        }

        @Override
        public Builder nOut(int nOut) {
            throw new UnsupportedOperationException("Unable to specify number of outputs with ocnn. Outputs are fixed to 1.");
        }

        @Override
        @SuppressWarnings("unchecked")
        public OCNNOutputLayer build() {
            return new OCNNOutputLayer(this);
        }
    }
}
