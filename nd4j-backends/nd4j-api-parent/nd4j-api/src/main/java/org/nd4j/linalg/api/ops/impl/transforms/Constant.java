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
                       boolean inPlace,int[] vertexId) {
        super();
        sameDiff.putShapeForVertexId(vertexId,shape);
        this.inPlace = inPlace;
        this.sameDiff = sameDiff;


        this.vertexId = vertexId;
        f().validateFunctionReference(this);
        if(sameDiff.getGraph().getVertex(this.vertexId[0]) == null) {
            sameDiff.getGraph().addVertex(new NDArrayVertex(sameDiff,vertexId[0],0,i_v));
        }

    }

    public Constant(SameDiff sameDiff,
                    SDVariable i_v,
                    int[] shape,int[] vertexId) {
        this(sameDiff,i_v,shape,false,vertexId);
    }

    /**
     * Get the result shape for this function
     *
     * @return
     */
    @Override
    public int[] getResultShape() {
        return sameDiff.getShapeForVertexId(vertexId);
    }

    @Override
    public boolean isConstant() {
        return true;
    }


    @Override
    public SDVariable getResult() {
        return sameDiff.getVariableForVertexId(vertexId);
    }

    @Override
    public DifferentialFunction[] args() {
        return new DifferentialFunction[] {this};
    }

    @Override
    public DifferentialFunction arg() {
        return this;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        f().validateDifferentialFunctionsameDiff(i_v);
        return Collections.<DifferentialFunction> singletonList(sameDiff.zero("grad-" + UUID.randomUUID().toString(),i_v.get(0).getResultShape()));

    }

    @Override
    public String toString() {
        return getResult().toString();
    }



    @Override
    public DifferentialFunction dup() {
        Constant ret = sameDiff.setupFunction(new Constant(sameDiff, sameDiff.getVariableForVertexId(vertexId),sameDiff.getShapeForVertexId(vertexId),vertexId));
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
