package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Collections;
import java.util.List;
import java.util.UUID;

@Data
public class Constant extends BaseTransformOp {


    public Constant() {
    }


    protected Constant(SameDiff sameDiff,
                       SDVariable i_v,
                       int[] shape,
                       boolean inPlace,int vertexId) {
        super();
        sameDiff.putShapeForVertexId(vertexId,shape);
        this.inPlace = inPlace;
        this.sameDiff = sameDiff;

        if(sameDiff.graph().getVertex(vertexId) == null) {
            sameDiff.graph().addVertex(new NDArrayVertex(sameDiff,vertexId,0,i_v));
        }

    }

    public Constant(SameDiff sameDiff,
                    SDVariable i_v,
                    int[] shape,int vertexId) {
        this(sameDiff,i_v,shape,false,vertexId);
    }
    @Override
    public boolean isConstant() {
        return true;
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(sameDiff.zero("grad-" + UUID.randomUUID().toString(),i_v.get(0).getShape()));

    }



    @Override
    public DifferentialFunction dup() {
        Constant ret = new Constant(sameDiff, sameDiff.getVariableForVertexId(outputVariables()[0].getVertexId())
                ,sameDiff.getShapeForVertexId(outputVariables()[0].getVertexId()),outputVariables()[0].getVertexId());
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
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
      throw new NoOpNameFoundException("No tensorflow opName found for " + calculateOutputShape());
    }

}
