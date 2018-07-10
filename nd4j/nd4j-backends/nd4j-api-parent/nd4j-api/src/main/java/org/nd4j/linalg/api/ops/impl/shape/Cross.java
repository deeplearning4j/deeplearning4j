package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;


/**
 * Pairwise cross-product of two tensors of the same shape.
 *
 * Base operation for two vectors is:
 *  a x b = (a_2 * b_3 - a_3 * b_2, a_3 * b_1 - a_1 * b_3, a_1 * b_2 - a_2 * b_1)
 *
 * For tensors of more complicated shape this op is computed pairwise. For this
 * to work the outer dimension has to be 3.
 *
 * @author Max Pumperla
 */
public class Cross extends DynamicCustomOp {

    public Cross() {
    }


    public Cross(SameDiff sameDiff, SDVariable[] args) {
        super(null, sameDiff, args, false);
    }

    public Cross(INDArray a, INDArray b, INDArray out){
        super(null, new INDArray[]{a,b}, new INDArray[]{out}, null, (int[])null);
    }

    @Override
    public String opName() {
        return "cross";
    }


    @Override
    public String tensorflowName() {
        return "Cross";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        /**
         * dL / dx = dL / dCross * dCross / dx
         * dCross(a,b) / da = Cross(1, b)
         * dCross(a,b) / db = Cross(a, 1)
         *
         * return (grad * Cross(1, b), grad * Cross(a, 1)
         */
        SDVariable grad = gradients.get(0);
        SDVariable a = larg();
        SDVariable b = rarg();
        SDVariable ones = sameDiff.onesLike(a);

        SDVariable gradLeft = grad.mul(sameDiff.cross(b, ones));
        SDVariable gradRight = grad.mul(sameDiff.cross(ones, a));

        return Arrays.asList(gradLeft, gradRight);
    }
}
