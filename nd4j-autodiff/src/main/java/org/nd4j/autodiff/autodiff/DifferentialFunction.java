package org.nd4j.autodiff.autodiff;

import java.util.List;

import org.nd4j.autodiff.Field;


public abstract class DifferentialFunction<X extends Field<X>>
        implements Field<DifferentialFunction<X>>, Differential<X, DifferentialFunction<X>> {

    protected DifferentialFunction() {
    }


    public abstract X getValue();

    @Override
    public abstract double getReal();

    public abstract String getFormula(List<Variable<X>> variables);

    @Override
    public abstract String toString();

    public boolean isPrecisionOK(int precision) {
        return (13 - precision) > Math.log10(getReal()) + 1;
    }


    public boolean isConstant() {
        return false;
    }


    public boolean isVariable() {
        return false;
    }

    @Override
    public abstract DifferentialFunction<X> diff(Variable<X> i_v1);

    @Override
    public DifferentialFunction<X> plus(DifferentialFunction<X> i_v) {
        return i_v.plused(this);
    }

    protected DifferentialFunction<X> plused(DifferentialFunction<X> i_v) {
        return new Sum<X>(i_v, this);
    }

    @Override
    public DifferentialFunction<X> minus(DifferentialFunction<X> i_v) {
        return plus(i_v.negate());
    }

    @Override
    public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
        return i_v.muled(this);
    }

    protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
        return new Product<X>(i_v, this);
    }

    @Override
    public DifferentialFunction<X> div(DifferentialFunction<X> i_v) {
        return mul(i_v.inverse());
    }

    @Override
    public DifferentialFunction<X> inverse() {
        return new Inverse<X>(this);
    }

    @Override
    public DifferentialFunction<X> negate() {
        return new Negative<X>(this);
    }

    @Override
    public DifferentialFunction<X> mul(long i_n) {
        return new PolynomialTerm<X>(i_n, this, 1);
    }

    @Override
    public DifferentialFunction<X> pow(int i_n) {
        return new PolynomialTerm<X>(1L, this, i_n);
    }

}
