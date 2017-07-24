package org.deeplearning4j.arbiter.adapter;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.adapter.ParameterSpaceAdapter;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A simple class to adapt a {@link LossFunctions.LossFunction} parameter space to a {@link ILossFunction} parameter space
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class LossFunctionParameterSpaceAdapter
                extends ParameterSpaceAdapter<LossFunctions.LossFunction, ILossFunction> {

    private ParameterSpace<LossFunctions.LossFunction> lossFunction;

    public LossFunctionParameterSpaceAdapter(
                    @JsonProperty("lossFunction") ParameterSpace<LossFunctions.LossFunction> lossFunction) {
        this.lossFunction = lossFunction;
    }

    @Override
    protected ILossFunction convertValue(LossFunctions.LossFunction from) {
        return from.getILossFunction();
    }

    @Override
    protected ParameterSpace<LossFunctions.LossFunction> underlying() {
        return lossFunction;
    }
}
