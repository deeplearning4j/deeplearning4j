package org.nd4j.autodiff.functions;

import java.util.List;

import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 *
 * @param <X>
 */
public class PolynomialTerm<X extends Field<X>> extends AbstractUnaryFunction<X> {

    protected double m_scale;
    protected int m_exponent;

    public PolynomialTerm(SameDiff sameDiff,
                          double i_scale,
                          DifferentialFunction<X> i_v,
                          int i_exponent,
                          boolean inPlace) {
        // scale v^{exponent}
        //note that super handles addEdges
        super(sameDiff,i_v,new Object[]{i_scale,i_exponent,inPlace});
        m_scale = i_scale;
        m_exponent = i_exponent;
    }

    public PolynomialTerm(SameDiff sameDiff,
                          double i_scale,
                          DifferentialFunction<X> i_v,
                          int i_exponent) {
        this(sameDiff,i_scale,i_v,i_exponent,false);
    }

    @Override
    public X doGetValue() {
        return (arg().getValue(true).pow(m_exponent)).mul(m_scale);
    }

    @Override
    public double getReal() {
        return Math.pow(arg().getReal(), m_exponent) * m_scale;
    }

    @Override
    public List<DifferentialFunction<X>> diff(List<DifferentialFunction<X>> i_v) {
        return (new PolynomialTerm<>(sameDiff,m_scale * m_exponent, arg(), m_exponent - 1))
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
    public String functionName() {
        return "pow";
    }

    @Override
    public DifferentialFunction<X> inverse() {
        return new PolynomialTerm<>(sameDiff, m_scale, arg(), -m_exponent);
    }

    @Override
    public DifferentialFunction<X> negate() {
        return new PolynomialTerm<>(sameDiff, -m_scale, arg(), m_exponent);
    }

}
