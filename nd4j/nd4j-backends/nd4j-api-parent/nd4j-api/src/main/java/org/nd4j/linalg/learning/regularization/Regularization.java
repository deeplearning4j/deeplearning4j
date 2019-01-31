package org.nd4j.linalg.learning.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface Regularization {

    /**
     * ApplyStep determines how the regularization interacts with the optimization process - i.e., when it is applied
     * relative to updaters like Adam, Nesterov momentum, SGD, etc.
     *
     *
     * BEFORE_UPDATER: w -= updater(gradient + regularization(p,gradView,lr)) <br>
     * POST_UPDATER: w -= (updater(gradient) + regularization(p,gradView,lr)) <br>
     * POST_PARAM_UPDATE: w -= updater(gradient); regularization(p, null, lr)) <br>
     *
     */
    enum ApplyStep {
        BEFORE_UPDATER,
        POST_UPDATER,
        POST_PARAM_UPDATE
    }

    ApplyStep applyStep();

    void apply(INDArray param, INDArray gradView, double lr);

    double score(INDArray param);

}
