package org.nd4j.autodiff.autodiff;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.Field;


public class Zero<X extends Field<X>> extends Constant<X> {


    public Zero(AbstractIdentityFactory<X> i_factory) {
        super(i_factory.zero(), i_factory);
    }

    @Override
    public DifferentialFunction<X> plus(DifferentialFunction<X> i_v) {
        return i_v;
    }

    @Override
    protected DifferentialFunction<X> plused(DifferentialFunction<X> i_v) {
        return i_v;
    }

    @Override
    // public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
    public Zero<X> mul(DifferentialFunction<X> i_v) {
        return this;
    }

    @Override
    // protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
    protected Zero<X> muled(DifferentialFunction<X> i_v) {
        return this;
    }

    @Override
    // public DifferentialFunction<X> inverse() {
    public Constant<X> inverse() {
        // TODO
        return null;
    }

    @Override
    // public DifferentialFunction<X> negate() {
    // public Constant<X> negate() {
    public Zero<X> negate() {
        return this;
    }

}
