package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * Constrain the weights to be non-negative
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class NonNegativeConstraint extends BaseConstraint {

    public NonNegativeConstraint(){ }

    @Override
    public void apply(INDArray param) {
        BooleanIndexing.replaceWhere(param, 0.0, Conditions.lessThan(0.0));
    }

    @Override
    public NonNegativeConstraint clone() { return new NonNegativeConstraint();}

}
