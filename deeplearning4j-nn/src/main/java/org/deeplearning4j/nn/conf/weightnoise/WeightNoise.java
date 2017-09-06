package org.deeplearning4j.nn.conf.weightnoise;

import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
public class WeightNoise implements IWeightNoise {

    private Distribution distribution;
    private boolean applyToBias;
    private boolean additive;

    public WeightNoise(Distribution distribution) {
        this(distribution, false, true);
    }

    public WeightNoise(Distribution distribution, boolean additive) {
        this(distribution, false, additive);
    }

    public WeightNoise(@JsonProperty("distribution") Distribution distribution,
                       @JsonProperty("applyToBias") boolean applyToBias,
                       @JsonProperty("additive") boolean additive) {
        this.distribution = distribution;
        this.applyToBias = applyToBias;
        this.additive = additive;
    }

    @Override
    public INDArray getParameter(Layer layer, String paramKey, int iteration, int epoch, boolean train) {

        ParamInitializer init = layer.conf().getLayer().initializer();
        INDArray param = layer.getParam(paramKey);
        if (train && init.isWeightParam(paramKey) || (applyToBias && init.isBiasParam(paramKey))) {

            org.nd4j.linalg.api.rng.distribution.Distribution dist = Distributions.createDistribution(distribution);
            INDArray noise = dist.sample(param.shape());
            INDArray out = Nd4j.createUninitialized(param.shape(), param.ordering());

            if (additive) {
                Nd4j.getExecutioner().exec(new AddOp(param, noise, out));
            } else {
                Nd4j.getExecutioner().exec(new MulOp(param, noise, out));
            }
            return out;
        }
        return param;
    }

    @Override
    public WeightNoise clone() {
        return new WeightNoise(distribution, applyToBias, additive);
    }
}
