package org.deeplearning4j.nn.conf.weightnoise;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.DropOut;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonProperty;


@Data
public class DropConnect implements IWeightNoise {

    private double weightRetainProbability;
    private boolean applyToBiases;

    public DropConnect(double weightRetainProbability) {
        this(weightRetainProbability, false);
    }

    public DropConnect(@JsonProperty("p") double weightRetainProbability, @JsonProperty("applyToBiases") boolean applyToBiases) {
        this.weightRetainProbability = weightRetainProbability;
        this.applyToBiases = applyToBiases;
    }

    @Override
    public INDArray getParameter(Layer layer, String paramKey, boolean train) {
        ParamInitializer init = layer.conf().getLayer().initializer();
        INDArray param = layer.getParam(paramKey);
        if (train && init.isWeightParam(paramKey) || (applyToBiases && init.isBiasParam(paramKey))) {
            INDArray out = Nd4j.createUninitialized(param.shape(), param.ordering());
            Nd4j.getExecutioner().exec(new DropOut(param, out, weightRetainProbability));
            return out;
        }
        return param;
    }

    @Override
    public DropConnect clone() {
        return new DropConnect(weightRetainProbability, applyToBiases);
    }
}
