package org.nd4j.autodiff.functions;


import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.UUID;

public abstract class AbstractScalarFunction extends AbstractUnaryFunction {
    protected Number scalarValue;

    public AbstractScalarFunction() {
        super();
    }

    public AbstractScalarFunction(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, Object[] extraArgs) {
        super(sameDiff, i_v, shape, OpState.OpType.SCALAR_TRANSFORM, extraArgs);
        this.scalarValue = (Number) extraArgs[0];
    }

    public AbstractScalarFunction(SameDiff sameDiff, DifferentialFunction i_v,boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v,i_v.getResultShape(), OpState.OpType.SCALAR_TRANSFORM,inPlace,extraArgs);
        this.scalarValue = (Number) extraArgs[0];
    }

    public AbstractScalarFunction(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
      this(sameDiff,i_v,false,extraArgs);
    }


    /**
     * Add nodes to the graph
     * @param sameDiff
     * @param i_v1
     * @param opName
     */
    @Override
    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            String opName,
                            int[] shape) {
        validateFunctionReference(i_v1);
        ArrayField v1 = i_v1.getValue(true);
        validateDifferentialFunctionsameDiff(v1);
        NDArrayInformation information =   inPlace ? i_v1.getResult() : NDArrayInformation.builder()
                .arrId(UUID.randomUUID().toString())
                .id(opName + "(" + v1.getInput().getId() + " -> " +
                        v1.getInput().getId() + ")")
                .shape(i_v1.getResultShape()).build();

        //result
        NDArrayVertex newVertex = new NDArrayVertex(
                sameDiff,
                sameDiff.graph().nextVertexId(),
                i_v1.getVertex().depth() + 1,
                information);
        this.vertexId = newVertex.vertexID();
        sameDiff.graph().addVertex(newVertex);


        OpState owner =  OpState.builder()
                .opType(OpState.OpType.SCALAR_TRANSFORM)
                .differentialFunction(this).inPlace(inPlace)
                .opName(opName).extraArgs(extraArgs).scalarValue((Number) extraArgs[0])
                .id(opName + "(" + v1.getInput().getId() + " -> " + newVertex.getValue().getId() + ")")
                .vertexIds(new String[]{String.valueOf(v1.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                .n(ArrayUtil.prod(shape)).scalarValue((Number) extraArgs[0])
                .result(information)
                .build();


        sameDiff.getGraph().addEdge(
                new int[]{arg().resultVertexId()},
                new int[]{newVertex.vertexID()},
                owner,
                true);


        newVertex.setOpState(owner);
        information.setOwner(owner);
        owner.setResult(information);

        if(owner.isInPlace()) {
            information.setArrId(v1.getInput().getArrId());
        }

        this.opState = owner;


    }


}
