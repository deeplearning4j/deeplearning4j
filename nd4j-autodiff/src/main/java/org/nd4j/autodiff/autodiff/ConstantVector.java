package org.nd4j.autodiff.autodiff;

import java.util.Collection;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.Field;

public class ConstantVector<X extends Field<X>> extends DifferentialVectorFunction<X> {

    public ConstantVector(AbstractIdentityFactory<X> i_factory, Constant<X>... i_v) {
        super(i_factory, i_v);
    }

    public ConstantVector(AbstractIdentityFactory<X> i_factory, Collection<Constant<X>> i_v) {
        super(i_factory, i_v);
    }

    public Constant<X> get(int i) {
        return (Constant<X>) m_v.get(i);
    }

}
