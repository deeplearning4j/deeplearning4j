package org.nd4j.autodiff.loss;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.autodiff.samediff.SDVariable;

/**
 * Information and variables for a loss function. Used with {@link LossFunctions}
 *
 * @author Alex Black
 */
@Builder(builderClassName = "Builder")
@Getter
public class LossInfo {
    private String lossName;
    private LossFunctions.Reduction reduction;
    private SDVariable loss;
    private SDVariable label;
    private SDVariable predictions;
    private SDVariable weights;

}
