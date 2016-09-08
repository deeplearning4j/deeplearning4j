package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Created by nyghtowl on 10/29/15.
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class LocalResponseNormalization extends Layer {
    // hyper-parameters defined using a validation set
    protected double n; // # adjacent kernal maps
    protected double k; // constant (e.g. scale)
    protected double beta; // decay rate
    protected double alpha; // decay rate

    private LocalResponseNormalization(Builder builder) {
        super(builder);
        this.k = builder.k;
        this.n = builder.n;
        this.alpha = builder.alpha;
        this.beta = builder.beta;
    }

    @Override
    public LocalResponseNormalization clone() {
        LocalResponseNormalization clone = (LocalResponseNormalization) super.clone();
        return clone;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization ret
                = new org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization(conf);
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
        return EmptyParamInitializer.getInstance();
    }


    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input type for LRN layer (layer name = \"" + getLayerName() + "\"): Expected input of type CNN, got " + inputType);
        }
        return inputType;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null ) {
            throw new IllegalStateException("Invalid input type for LRN layer (layer name = \"" + getLayerName() + "\"): null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public double getL1ByParam(String paramName) {
        //Not applicable
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        //Not applicable
        return 0;
    }

    @Override
    public double getLearningRateByParam(String paramName) {
        //Not applicable
        return 0;
    }

    @AllArgsConstructor
    public static class Builder extends Layer.Builder<Builder> {
        // defaults based on AlexNet model
        private double k = 2;
        private double n = 5;
        private double alpha = 1e-4;
        private double beta = 0.75;

        public Builder(double k, double alpha, double beta) {
            this.k = k;
            this.alpha = alpha;
            this.beta = beta;
        }

        public Builder() {
        }

        public Builder k(double k) {
            this.k = k;
            return this;
        }

        public Builder n(double n) {
            this.n = n;
            return this;
        }

        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder beta(double beta) {
            this.beta = beta;
            return this;
        }

        @Override
        public LocalResponseNormalization build() {
            return new LocalResponseNormalization(this);
        }

    }

}
