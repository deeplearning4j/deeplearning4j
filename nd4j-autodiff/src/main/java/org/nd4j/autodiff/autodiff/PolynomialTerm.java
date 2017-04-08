package org.nd4j.autodiff.autodiff;

import java.util.List;

import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;

public class PolynomialTerm<X extends Field<X>> extends AbstractUnaryFunction<X> {

    protected long m_scale;
    protected int m_exponent;

    public PolynomialTerm(Graph<NDArrayInformation,OpState> graph, long i_scale, DifferentialFunction<X> i_v, int i_exponent) {
        // scale v^{exponent}
        super(graph,i_v);
        m_scale = i_scale;
        m_exponent = i_exponent;
    }

    @Override
    public X doGetValue() {
        return (arg().getValue().pow(m_exponent)).mul(m_scale);
    }

    @Override
    public double getReal() {
        return Math.pow(arg().getReal(), m_exponent) * m_scale;
    }

    @Override
    public DifferentialFunction<X> diff(Variable<X> i_v) {
        return (new PolynomialTerm<>(graph,m_scale * m_exponent, arg(), m_exponent - 1))
                .mul(arg().diff(i_v));
    }

    @Override
    public String toString() {
        return m_scale + arg().toString() + "^" + m_exponent;
    }

    @Override
    public String doGetFormula(List<Variable<X>> variables) {
        return "( " + m_scale + " * Math.pow(" + arg().doGetFormula(variables) + "," + m_exponent
                + ")";
    }

    @Override
    public DifferentialFunction<X> inverse() {
        return new PolynomialTerm<X>(graph,m_scale, arg(), -m_exponent);
    }

    @Override
    public DifferentialFunction<X> negate() {
        return new PolynomialTerm<X>(graph,-m_scale, arg(), m_exponent);
    }

}
