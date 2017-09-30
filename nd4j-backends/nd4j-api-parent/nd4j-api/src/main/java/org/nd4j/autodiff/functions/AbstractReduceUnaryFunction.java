package org.nd4j.autodiff.functions;

import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

@Data
public abstract class AbstractReduceUnaryFunction extends DifferentialFunction {

    protected DifferentialFunction m_x;
    protected int[] dimensions;

    public AbstractReduceUnaryFunction(SameDiff sameDiff,
                                       DifferentialFunction i_v,
                                       int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            m_x = i_v;
            this.dimensions = dimensions;
            validateDifferentialFunctionsameDiff(i_v);

            addEdges(sameDiff,m_x,functionName());
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }

    @Override
    public ArrayField doGetValue() {
        return a().prod(arg().getValue(true),dimensions);
    }


    @Override
    public String toString() {
        return functionName() + "(" + m_x + ",axes:" + Arrays.toString(dimensions) + ")";
    }

    @Override
    public String doGetFormula(List<Variable> variables) {
        return functionName() + "(" + m_x.doGetFormula(new ArrayList<Variable>()) + ",axes:" + Arrays.toString(dimensions) + ")";
    }

    /**
     * Add nodes to the graph
     * @param sameDiff
     * @param i_v1
     * @param opName
     */
    protected void addEdges(SameDiff sameDiff, DifferentialFunction i_v1,String opName) {
        ArrayField v1 = i_v1.getValue(true);
        int[] resultShape = Shape.getReducedShape(v1.getInput().getShape(),dimensions);
        //result
        NDArrayInformation information =  NDArrayInformation.builder()
                .arrId(UUID.randomUUID().toString())
                .id(opName + "(" + v1.getInput().getId() + " -> " + v1.getInput().getId() + ")")
                .shape(resultShape).build();
        NDArrayVertex newVertex = new NDArrayVertex(sameDiff,
                sameDiff.getGraph().nextVertexId(),
                i_v1.getVertex().depth() + 1,
                information);
        this.vertexId = newVertex.vertexID();
        sameDiff.getGraph().addVertex(newVertex);
        OpState opState =   OpState.builder()
                .opType(OpState.OpType.ACCUMULATION)
                .opName(opName).axes(dimensions)
                .id(opName + "(" + v1.getInput().getId() + " -> " + newVertex.getValue().getId() + ")")
                .vertexIds(sameDiff.generateVertexIds(v1.getVertex().vertexID(),newVertex.vertexID()))
                .n(ArrayUtil.prod(Shape.getReducedShape(v1.getInput().getShape(),dimensions)))
                .differentialFunction(this)
                .build();
        newVertex.setOpState(opState);
        sameDiff.getGraph().addEdge(
                new int[]{v1.getVertex().vertexID()}
        ,new int[]{newVertex.vertexID()},opState
                ,true);
        this.opState = opState;
        information.setOwner(opState);
        opState.setResult(information);


    }

    @Override
    public DifferentialFunction[] args() {
        return new DifferentialFunction[] {m_x};
    }

    public DifferentialFunction arg() {
        return m_x;
    }


    @Override
    public DifferentialFunction dup() {
        try {
            return getClass().getConstructor(sameDiff.getClass(),arg()
                    .getClass(),int[].class).newInstance(sameDiff,arg(),dimensions);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
