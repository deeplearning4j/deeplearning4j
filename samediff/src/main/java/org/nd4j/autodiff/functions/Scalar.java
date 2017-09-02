package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;


/**
 * Scalar value
 *
 */
public class Scalar extends Constant {

    protected double value;

    public Scalar(SameDiff sameDiff,
                  double value) {
        this(sameDiff, value, false);

    }

    public Scalar(SameDiff sameDiff,
                  double value,boolean inPlace) {
        super(sameDiff,  sameDiff.getArrayFactory().scalar(value),new int[]{1,1},inPlace);
        this.value = value;

    }


    @Override
    public DifferentialFunction<ArrayField> mul(DifferentialFunction<ArrayField> i_v) {
        DifferentialFunction<ArrayField> dup = i_v.dup();
        ArrayField arrayField = i_v.getValue(true);
        addEdges(sameDiff,
                dup,
                this,
                new MulOp().name(),
                OpState.OpType.TRANSFORM,
                arrayField.getInput().getShape());


        return dup;
    }


    @Override
    public DifferentialFunction<ArrayField> dup() {
        return new Scalar(sameDiff, value);
    }
}
