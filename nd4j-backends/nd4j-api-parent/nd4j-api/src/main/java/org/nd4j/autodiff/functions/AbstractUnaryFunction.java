package org.nd4j.autodiff.functions;

import com.google.common.base.Preconditions;
import com.rits.cloning.Cloner;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.List;
import java.util.UUID;

@Data
@NoArgsConstructor
public abstract class AbstractUnaryFunction extends DifferentialFunction {

    protected DifferentialFunction m_x;
    protected int[] shape;
    protected OpState.OpType opType;

    public AbstractUnaryFunction(SameDiff sameDiff,
                                 DifferentialFunction i_v,
                                 int[] shape,
                                 OpState.OpType opType,
                                 Object[] extraArgs) {
        this(sameDiff,
                i_v,
                shape,
                opType,
                false,
                extraArgs);
    }

    public AbstractUnaryFunction(SameDiff sameDiff,DifferentialFunction i_v,boolean inPlace) {
        this(sameDiff,i_v,i_v.getResultShape(), OpState.OpType.TRANSFORM,inPlace,null);
    }

    public AbstractUnaryFunction(SameDiff sameDiff,
                                 DifferentialFunction i_v,
                                 int[] shape,
                                 OpState.OpType opType,
                                 boolean inPlace,
                                 Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        this.opType = opType;
        this.shape = shape;

        if (i_v != null) {
            m_x = sameDiff.setupFunction(i_v);
            validateFunctionReference(i_v);
            validateDifferentialFunctionsameDiff(i_v);
            addEdges(sameDiff,m_x,functionName(),shape);
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }

    public AbstractUnaryFunction(SameDiff sameDiff,
                                 DifferentialFunction i_v,
                                 int[] shape,
                                 Object[] extraArgs) {
        this(sameDiff,i_v,shape, OpState.OpType.TRANSFORM,extraArgs);
    }


    public AbstractUnaryFunction(SameDiff sameDiff,
                                 DifferentialFunction i_v,
                                 Object[] extraArgs) {
        this(sameDiff,i_v,i_v.getResultShape(), OpState.OpType.TRANSFORM,extraArgs);
    }


    @Override
    public String doGetFormula(List<Variable> variables) {
        return functionName() + "(" + arg().doGetFormula(variables) + ")";
    }

    @Override
    public String toString() {
        return functionName() + "(" + arg().toString() + ")";
    }

    /**
     * Add nodes to the graph
     * @param sameDiff
     * @param i_v1
     * @param opName
     */
    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            String opName,
                            int...shape) {
        validateFunctionReference(i_v1);
        ArrayField v1 = i_v1.getValue(true);
        validateDifferentialFunctionsameDiff(v1);
        NDArrayInformation information =   inPlace ? i_v1.getResult() :  NDArrayInformation.builder()
                .arrId(UUID.randomUUID().toString())
                .id(opName + "(" + v1.getInput().getId() + " -> " +
                        v1.getInput().getId() + ")")
                .shape(shape).build();
        //result
        NDArrayVertex newVertex = new NDArrayVertex(
                sameDiff,
                sameDiff.graph().nextVertexId(),
                i_v1.getVertex().depth() + 1,
                information);
        this.vertexId = newVertex.vertexID();
        sameDiff.graph().addVertex(newVertex);
        Preconditions.checkArgument(sameDiff == i_v1.sameDiff,"Illegal samediff instance");
        OpState owner =  OpState.builder()
                .opType(opType).differentialFunction(this)
                .opName(opName).inPlace(inPlace)
                .extraArgs(extraArgs)
                .id(opName + "(" + v1.getInput().getId() + " -> " + newVertex.getValue().getId() + ")")
                .vertexIds(sameDiff.generateVertexIds(v1.getVertex().vertexID(),newVertex.vertexID()))
                .n(ArrayUtil.prod(shape)).result(information)
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


    @Override
    public DifferentialFunction[] args() {
        return new DifferentialFunction[] {arg()};
    }

    @Override
    public DifferentialFunction arg() {
        return m_x;
    }


    @Override
    public DifferentialFunction dup() {
        Cloner cloner = new Cloner();
        return sameDiff.setupFunction(cloner.deepClone(this));
    }


}

