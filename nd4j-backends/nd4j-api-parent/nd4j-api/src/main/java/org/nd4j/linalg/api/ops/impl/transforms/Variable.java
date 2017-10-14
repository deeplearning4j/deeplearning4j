package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.Data;
import lombok.Getter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@Data
public class Variable extends DifferentialFunction {
    @Getter
    private NDArrayInformation m_x;
    private String m_name;

    public Variable(SameDiff sameDiff,
                    String i_name,
                    NDArrayInformation i_v,int vertexId) {
        super(sameDiff,null);
        setName(i_name);
        if (i_v != null) {
            m_x = i_v;
            validateFunctionReference(this);
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }

        this.vertexId = vertexId;
        if(sameDiff.graph().getVertex(vertexId) == null) {
            sameDiff.graph().addVertex(new NDArrayVertex(sameDiff,vertexId,0,i_v));
        }

        if(sameDiff.getFunctionInstances().get(vertexId) == null) {
            sameDiff.getFunctionInstances().put(vertexId,this);
        }

    }


    public Variable(SameDiff sameDiff,
                    String i_name,
                    NDArrayInformation i_v) {
        this(sameDiff,i_name,i_v,sameDiff.graph().nextVertexId());

    }

    @Override
    public int depth() {
        return sameDiff.graph().getVertex(vertexId).depth();
    }

    @Override
    public int[] getOutputVertexIds() {
        return new int[] {vertexId};
    }


    @Override
    public List<DifferentialFunction> outputs() {
        return Collections.singletonList((DifferentialFunction) this);
    }

    private void setName(String i_name) {
        if (i_name != null) {
            m_name = i_name;// new String(i_name);
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }
    }

    public String getName() {
        return m_name;
    }

    public void set(NDArrayInformation i_v) {
        if (i_v != null) {
            m_x = i_v;
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }
    }

    @Override
    public boolean isVariable() {
        return true;
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
        //default value is 1.0 (constant)
        List<DifferentialFunction> ret = new ArrayList<>();
        if(i_v == this)
            ret.add(sameDiff.setupFunction(f().one(i_v.get(0)
                    .getResultShape())));
        else
            ret.add(sameDiff.setupFunction(f().zero(i_v.get(0).getResultShape())));
        return ret;


    }

    @Override
    public NDArrayInformation getResult() {
        return m_x;
    }

    /**
     * Get the result shape for this function
     * @return
     */
    @Override
    public int[] getResultShape() {
        NDArrayInformation arrayField = m_x;
        return arrayField.getShape();
    }


    @Override
    public String doGetFormula(List<Variable> variables) {
        variables.add(this);
        return getName();
    }

    @Override
    public DifferentialFunction dup() {
        return sameDiff.setupFunction(new Variable(sameDiff, getName(),
                m_x,vertexId));
    }

    @Override
    public String toString() {
        return "Variable{" +
                "m_name='" + m_name + '\'' +
                ", vertexId=" + vertexId +
                ", extraArgs=" + Arrays.toString(extraArgs) +
                '}';
    }
}
