package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;

import java.util.UUID;


public class Zero extends Constant {
    public Zero() {
    }

    public Zero(SameDiff sameDiff, int[] shape,int[] vertexId) {
        super(sameDiff, SDVariable
                .builder()
                .varName("zero")
                .shape(shape)
                .build(),shape,vertexId);
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
