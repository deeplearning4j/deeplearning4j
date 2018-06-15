package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

@Data
public class Constant extends BaseTransformOp {


    public Constant() {
    }


    protected Constant(SameDiff sameDiff,
                       SDVariable i_v,
                       long[] shape,
                       boolean inPlace) {
        super();
        sameDiff.putShapeForVarName(i_v.getVarName(), shape);
        this.xVertexId = i_v.getVarName();
        this.inPlace = inPlace;
        this.sameDiff = sameDiff;
    }

    public Constant(SameDiff sameDiff, SDVariable i_v, long[] shape) {
        this(sameDiff, i_v, shape, false);
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }


    @Override
    public DifferentialFunction dup() {
        Constant ret = new Constant(sameDiff, sameDiff.getVariable(outputVariables()[0].getVarName())
                , sameDiff.getShapeForVarName(outputVariables()[0].getVarName()));
        Constant differentialFunction = ret;
        return differentialFunction;
    }


    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "constant";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow opName found for " + opName());
    }

}
