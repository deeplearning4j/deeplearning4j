package org.nd4j.autodiff.functions;

import java.util.Arrays;
import java.util.List;

import com.google.common.base.Preconditions;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.samediff.SameDiff;

@Data
public class Constant extends DifferentialFunction {

    protected ArrayField m_x;
    protected int[] shape;

    protected Constant(SameDiff sameDiff,
                       ArrayField i_v,
                       int[] shape,
                       boolean inPlace) {
        super(sameDiff,new Object[]{i_v,inPlace});
        this.shape = shape;
        if (i_v != null) {
            m_x = i_v;

        } else {
            throw new IllegalArgumentException("Input not null value.");
        }

        ArrayField arrayField = i_v;
        validateDifferentialFunctionsameDiff(arrayField);
        this.vertexId = arrayField.getVertex().vertexID();
        validateFunctionReference(this);
        if(sameDiff.getGraph().getVertex(this.vertexId) == null)
            sameDiff.getGraph().addVertex(arrayField.getVertex());


    }

    protected Constant(SameDiff sameDiff,
                       ArrayField i_v,
                       int[] shape) {
        this(sameDiff,i_v,shape,false);
    }



    /**
     * Get the result shape for this function
     *
     * @return
     */
    @Override
    public int[] getResultShape() {
        return shape;
    }

    @Override
    public boolean isConstant() {
        return true;
    }

    @Override
    public ArrayField doGetValue() {
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
        validateDifferentialFunctionsameDiff(i_v);
        Zero ret = new Zero(sameDiff,shape);
        DifferentialFunction add = ret;
        return Arrays.asList(add);
    }

    @Override
    public String toString() {
        return getValue(true).toString();
    }

    @Override
    public String doGetFormula(List<Variable> variables) {
        return getValue(true).toString();
    }

    @Override
    public String functionName() {
        return "constant";
    }


    @Override
    public DifferentialFunction dup() {
        Constant ret = sameDiff.setupFunction(new Constant(sameDiff, m_x, shape));
        Constant differentialFunction = ret;
        return differentialFunction;
    }

    // This class must be immutable.
    // set and assign must not be implemented.
    @SuppressWarnings("unused")
    private final void set(ArrayField i_x) {
    }

    @SuppressWarnings("unused")
    private final void assign(ArrayField i_x) {
    }

}
