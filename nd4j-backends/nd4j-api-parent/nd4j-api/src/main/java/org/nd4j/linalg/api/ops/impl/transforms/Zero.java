package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.samediff.SameDiff;


public class Zero extends Constant {
    public Zero() {
    }

    public Zero(SameDiff sameDiff, int[] shape,int[] vertexId) {
        super(sameDiff, NDArrayInformation.newInfo(shape),shape,vertexId);
    }


    @Override
    public String name() {
        return "zero";
    }

    @Override
    public DifferentialFunction dup() {
        return new Zero(sameDiff,shape,vertexId);
    }
}
