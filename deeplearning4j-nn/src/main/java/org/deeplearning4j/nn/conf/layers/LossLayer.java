package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.util.Collection;
import java.util.Map;

/**
 * LossLayer is a flexible output "layer" that performs a loss function on
 * an input without MLP logic.
 *
 * @author Justin Long (crockpotveggies)
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class LossLayer extends FeedForwardLayer {
    protected ILossFunction lossFn;

    protected LossLayer(Builder builder) {
        super(builder);
        this.lossFn = builder.lossFn;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.LossLayer ret
            = new org.deeplearning4j.nn.layers.LossLayer(conf);
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

    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        public Builder() {
            this.activation("identity");
        }

        public Builder(LossFunctions.LossFunction lossFunction) {
            lossFunction(lossFunction);
            this.activation("identity");
        }

        public Builder(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
            this.activation("identity");
        }

        @Override
        @SuppressWarnings("unchecked")
        public Builder nIn(int nIn) {
            throw new UnsupportedOperationException("Ths layer has no parameters, thus nIn will always equal nOut.");
        }

        @Override
        @SuppressWarnings("unchecked")
        public Builder nOut(int nOut) {
            throw new UnsupportedOperationException("Ths layer has no parameters, thus nIn will always equal nOut.");
        }

        @Override
        @SuppressWarnings("unchecked")
        public LossLayer build() {
            return new LossLayer(this);
        }
    }
}
