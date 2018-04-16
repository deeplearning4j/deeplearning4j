package org.deeplearning4j.nn.conf.weightnoise;

import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldAddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Apply noise of the specified distribution to the weights at training time.
 * Note that both additive and multiplicative modes are supported - when additive, noise should be mean 0,
 * when multiplicative, noise should be mean 1.
 * That is, additive noise: x = x + noise<br>
 * multiplicative noise: x = x * noise
 *
 * @author Alex Black
 */
@Data
public class WeightNoise implements IWeightNoise {

    private Distribution distribution;
    private boolean applyToBias;
    private boolean additive;

    /**
     * @param distribution Distribution for additive noise
     */
    public WeightNoise(Distribution distribution) {
        this(distribution, false, true);
    }

    /**
     * @param distribution Distribution for noise
     * @param additive     If true: noise is added to weights. If false: noise is multiplied by weights
     */
    public WeightNoise(Distribution distribution, boolean additive) {
        this(distribution, false, additive);
    }

    /**
     * @param distribution Distribution for noise
     * @param applyToBias  If true: apply to biases also. If false (default): apply only to weights
     * @param additive     If true: noise is added to weights. If false: noise is multiplied by weights
     */
    public WeightNoise(@JsonProperty("distribution") Distribution distribution,
                       @JsonProperty("applyToBias") boolean applyToBias,
                       @JsonProperty("additive") boolean additive) {
        this.distribution = distribution;
        this.applyToBias = applyToBias;
        this.additive = additive;
    }

    @Override
    public INDArray getParameter(Layer layer, String paramKey, int iteration, int epoch, boolean train, LayerWorkspaceMgr workspaceMgr) {

        ParamInitializer init = layer.conf().getLayer().initializer();
        INDArray param = layer.getParam(paramKey);
        if (train && init.isWeightParam(layer.conf().getLayer(), paramKey) ||
                (applyToBias && init.isBiasParam(layer.conf().getLayer(), paramKey))) {

            org.nd4j.linalg.api.rng.distribution.Distribution dist = Distributions.createDistribution(distribution);
            INDArray noise = dist.sample(param.shape());
            INDArray out = workspaceMgr.createUninitialized(ArrayType.INPUT, param.shape(), param.ordering());

            if (additive) {
                Nd4j.getExecutioner().exec(new OldAddOp(param, noise,out));
            } else {
                Nd4j.getExecutioner().exec(new OldMulOp(param, noise, out));
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
