package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.Data;
import lombok.Getter;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.PreEvaluator;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Data
public class Variable extends DifferentialFunction {
    @Getter
    private ArrayField m_x;
    private String m_name;
    private PreEvaluator preEvaluator;

    public Variable(SameDiff sameDiff,
                       String i_name,
                       ArrayField i_v) {
        this(sameDiff,i_name, i_v, null);
        validateFunctionReference(this);
    }

    public Variable(SameDiff sameDiff,
                       String i_name,
                       ArrayField i_v,PreEvaluator preEvaluator) {
        super(sameDiff,null);
        this.preEvaluator = preEvaluator;
        setName(i_name);
        if (i_v != null) {
            m_x = i_v;
            validateFunctionReference(this);
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }

        ArrayField arrayField = i_v;
        validateDifferentialFunctionsameDiff(arrayField);
        this.vertexId = arrayField.getVertex().vertexID();


    }


    /**
     * Get the value specifying
     * whether to freeze the graph or not
     * @param freeze whether to freeze the graph or not,
     *               this means whether to add nodes to the internal
     *               computation graph or not
     * @return the value of this function
     */
    @Override
    public  ArrayField getValue(boolean freeze) {
        if(freeze) {
            return m_x;
        }

        return super.getValue(freeze);
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

    public void set(ArrayField i_v) {
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
    public ArrayField doGetValue() {
        if (preEvaluator != null) {
            preEvaluator.update(this);
        }
        return m_x;
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
        return m_x.getInput();
    }

    /**
     * Get the result shape for this function
     * @return
     */
    @Override
    public int[] getResultShape() {
        ArrayField arrayField = m_x;
        return arrayField.getInput().getShape();
    }


    @Override
    public String doGetFormula(List<Variable> variables) {
        variables.add(this);
        return getName();
    }

    @Override
    public DifferentialFunction dup() {
        return sameDiff.setupFunction(new Variable(sameDiff, getName(),
                m_x));
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
