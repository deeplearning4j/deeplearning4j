package org.nd4j.autodiff.functions;

import java.util.List;

import lombok.Data;
import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.samediff.SameDiffGraph;

@Data
public class Constant<X extends Field<X>> extends DifferentialFunction<X> {

    protected X m_x;
    protected AbstractIdentityFactory<X> m_factory;
    protected int[] shape;

    protected Constant(SameDiffGraph graph,
                       X i_v,
                       int[] shape,
                       AbstractIdentityFactory<X> i_factory) {
        super(graph,new Object[]{i_v});
        this.shape = shape;
        if(i_factory == null) {
            i_factory = (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory();
        }
        if (i_v != null && i_factory != null) {
            m_x = i_v;
            m_factory = i_factory;

        } else {
            throw new IllegalArgumentException("Input not null value.");
        }

        if(i_v instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) i_v;
            this.vertexId = arrayField.getVertex().vertexID();
        }
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
    public DifferentialFunction<X> diff(Variable<X> i_v) {
        return new Zero<>(graph,shape, m_factory);
    }

    @Override
    public String toString() {
        return getValue().toString();
    }

    @Override
    public String doGetFormula(List<Variable<X>> variables) {
        return getValue().toString();
    }

    @Override
    public String functionName() {
        return "constant";
    }



    @Override
    public Constant<X> inverse() {
        Constant<X> ret = new Constant<>(graph, m_x.inverse(),shape, m_factory);
        return ret;
    }

    @Override
    public Constant<X> negate() {
        Constant<X> ret =  new Constant<>(graph, m_x.negate(),shape, m_factory);
        return ret;
    }

    @Override
    public DifferentialFunction<X> dup() {
        return new Constant<>(graph,m_x,shape,getM_factory());
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
