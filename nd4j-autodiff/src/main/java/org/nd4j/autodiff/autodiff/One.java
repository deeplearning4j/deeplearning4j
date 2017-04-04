package org.nd4j.autodiff.autodiff;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.Field;


public class One<X extends Field<X>> extends Constant<X> {


    public One(AbstractIdentityFactory<X> i_factory) {
        super(i_factory.one(), i_factory);
    }

    public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
        return i_v;
    }

    protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
        return i_v;
    }

}
