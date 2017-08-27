package org.nd4j.autodiff.functions;

import java.util.List;

import com.google.common.base.Preconditions;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.samediff.SameDiff;

@Data
public class Constant<X extends Field<X>> extends DifferentialFunction<X> {

    protected X m_x;
    protected int[] shape;

    protected Constant(SameDiff sameDiff,
                       X i_v,
                       int[] shape,
                       boolean inPlace) {
        super(sameDiff,new Object[]{i_v,inPlace});
        this.shape = shape;
        if (i_v != null) {
            m_x = i_v;

        } else {
            throw new IllegalArgumentException("Input not null value.");
        }

        if(i_v instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) i_v;
            Preconditions.checkArgument(arrayField.getOps() == sameDiff,"Constant instantiated with wrong samediff from arrayfield.");
            this.vertexId = arrayField.getVertex().vertexID();
            if(sameDiff.getGraph().getVertex(this.vertexId) == null)
                sameDiff.getGraph().addVertex(arrayField.getVertex());

        }
    }

    protected Constant(SameDiff sameDiff,
                       X i_v,
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
    public X doGetValue() {
        return m_x;
    }

    @Override
    public double getReal() {
        return m_x.getReal();
    }

    @Override
    public DifferentialFunction<X>[] args() {
        return new DifferentialFunction[] {this};
    }

    @Override
    public DifferentialFunction<X> arg() {
        return this;
    }

    @Override
    public List<DifferentialFunction<X>> diff(List<DifferentialFunction<X>> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return new Zero<>(sameDiff,shape);
    }

    @Override
    public String toString() {
        return getValue(true).toString();
    }

    @Override
    public String doGetFormula(List<Variable<X>> variables) {
        return getValue(true).toString();
    }

    @Override
    public String functionName() {
        return "constant";
    }



    @Override
    public Constant<X> inverse() {
        Constant<X> ret = new Constant<>(sameDiff, m_x.inverse(),shape);
        return ret;
    }

    @Override
    public Constant<X> negate() {
        Constant<X> ret =  new Constant<>(sameDiff, m_x.negate(),shape);
        validateDifferentialFunctionsameDiff(ret);
        return ret;
    }

    @Override
    public DifferentialFunction<X> dup() {
        return new Constant<>(sameDiff,m_x,shape);
    }

    // This class must be immutable.
    // set and assign must not be implemented.
    @SuppressWarnings("unused")
    private final void set(X i_x) {
    }

    @SuppressWarnings("unused")
    private final void assign(X i_x) {
    }

}
