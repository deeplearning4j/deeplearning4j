package org.nd4j.autodiff.autodiff;

import java.util.List;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;


public class Constant<X extends Field<X>> extends DifferentialFunction<X> {

    private X m_x;
    private AbstractIdentityFactory<X> m_factory;

    protected Constant(Graph<NDArrayInformation,OpState> graph, X i_v, AbstractIdentityFactory<X> i_factory) {
        super(graph);
        if (i_v != null && i_factory != null) {
            m_x = i_v;
            m_factory = i_factory;
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }
    }

    protected AbstractIdentityFactory<X> factory() {
        return m_factory;
    }

    @Override
    public boolean isConstant() {
        return true;
    }

    @Override
    public X getValue() {
        return m_x;
    }

    @Override
    public double getReal() {
        return m_x.getReal();
    }

    @Override
    public DifferentialFunction<X> diff(Variable<X> i_v) {
        return new Zero<X>(graph,m_factory);
    }

    @Override
    public String toString() {
        return getValue().toString();
    }

    @Override
    public String getFormula(List<Variable<X>> variables) {
        return getValue().toString();
    }

    @Override
    protected DifferentialFunction<X> plused(DifferentialFunction<X> i_v) {
        return i_v.isConstant() ? new Constant<>(graph, i_v.getValue().plus(this.m_x), m_factory)
                : super.plused(i_v);
    }

    @Override
    protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
        return i_v.isConstant() ? new Constant<>(graph, i_v.getValue().mul(this.m_x), m_factory)
                : super.muled(i_v);
    }

    // public DifferentialFunction<X> inverse() {
    @Override
    public Constant<X> inverse() {
        return new Constant<>(graph, m_x.inverse(), m_factory);
    }

    // public DifferentialFunction<X> negate() {
    @Override
    public Constant<X> negate() {
        return new Constant<>(graph, m_x.negate(), m_factory);
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
