package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.Constant;


/**
 * Scalar value
 *
 */
public class Scalar extends Constant {

    protected double value;

    public Scalar(SameDiff sameDiff,
                  double value,int[] vertexId) {
        this(sameDiff, value, false,vertexId);

    }

    public Scalar(SameDiff sameDiff,
                  double value,boolean inPlace,int[] vertexId) {
        super(sameDiff, NDArrayInformation.newInfo(new int[]{1,1}),new int[]{1,1},inPlace,vertexId);
        this.value = value;

    }



    @Override
    public DifferentialFunction dup() {
        return new Scalar(sameDiff, value,vertexId);
    }
}
