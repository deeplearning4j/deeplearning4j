package org.nd4j.autodiff.functions;

import java.util.List;

import lombok.Data;
import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.util.ArrayUtil;

@Data
public class Constant<X extends Field<X>> extends DifferentialFunction<X> {

    protected X m_x;
    private AbstractIdentityFactory<X> m_factory;

    protected Constant(Graph<NDArrayInformation,OpState> graph,
                       X i_v,
                       AbstractIdentityFactory<X> i_factory) {
        super(graph,new Object[]{i_v});
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
        return new Zero<>(graph, m_factory);
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
    protected DifferentialFunction<X> plused(DifferentialFunction<X> i_v) {
        return i_v.isConstant() ? new Constant<>(graph, i_v.getValue(false).plus(this.m_x), m_factory)
                : super.plused(i_v);
    }

    @Override
    protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
        return i_v.isConstant() ? new Constant<>(graph, i_v.getValue(false).mul(this.m_x), m_factory)
                : super.muled(i_v);
    }


    @Override
    public Constant<X> inverse() {
        Constant<X> ret = new Constant<>(graph, m_x.inverse(), m_factory);
        return ret;
    }

    @Override
    public Constant<X> negate() {
        Constant<X> ret =  new Constant<>(graph, m_x.negate(), m_factory);
        return ret;
    }

    @Override
    public DifferentialFunction<X> dup() {
        return new Constant<>(graph,m_x,getM_factory());
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
