package org.nd4j.autodiff.autodiff;

import java.util.ArrayList;

import java.util.Collection;
import java.util.List;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.BasicDenseVector;
import org.nd4j.autodiff.CommutativeGroup;
import org.nd4j.autodiff.Field;


public class DifferentialVectorFunction<X extends Field<X>>
        implements CommutativeGroup<DifferentialVectorFunction<X>>,
        Differential<X, DifferentialVectorFunction<X>> { // ,
                                                         // VectorDifferential<X,
                                                         // DifferentialMatrixFunction<X>>
                                                         // {

    protected AbstractIdentityFactory<X> m_factory;
    protected BasicDenseVector<DifferentialFunction<X>> m_v;

    public DifferentialVectorFunction(AbstractIdentityFactory<X> i_factory,
            DifferentialFunction<X>... i_v) {
        m_factory = i_factory;
        m_v = new BasicDenseVector<>(i_v);
    }

    public DifferentialVectorFunction(AbstractIdentityFactory<X> i_factory,
            Collection<? extends DifferentialFunction<X>> i_v) {
        m_factory = i_factory;
        m_v = new BasicDenseVector<>(i_v);
    }

    // protected DifferentialVectorFunction(){
    // }
    protected DifferentialVectorFunction(AbstractIdentityFactory<X> i_factory,
            BasicDenseVector<DifferentialFunction<X>> i_v) {
        m_factory = i_factory;
        m_v = i_v;
    }

    public int size() {
        return m_v.size();
    }

    public DifferentialFunction<X> get(int i) {
        return m_v.get(i);
    }

    public DifferentialVectorFunction<X> diff(Variable<X> i_v) {
        int size = this.size();
        List<DifferentialFunction<X>> v = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            v.add(this.get(i).diff(i_v));
        }
        return new DifferentialVectorFunction<>(m_factory, v);
    }

    public DifferentialFunction<X> dot(DifferentialVectorFunction<X> i_v) {
        int size = this.size();
        if (i_v.size() != size) {
            // throw Error
            return null;
        }
        DifferentialFunction<X> norm = new Zero<X>(m_factory);
        for (int i = 0; i < size; i++) {
            norm = norm.plus(this.get(i).mul(i_v.get(i)));
        }
        return norm;
    }

    public DifferentialVectorFunction<X> negate() {
        return new DifferentialVectorFunction<X>(m_factory, m_v.negate());
    }

    public DifferentialVectorFunction<X> plus(DifferentialVectorFunction<X> i_v) {
        return new DifferentialVectorFunction<>(m_factory, m_v.plus(i_v.m_v));
    }

    public DifferentialVectorFunction<X> minus(DifferentialVectorFunction<X> i_v) {
        return new DifferentialVectorFunction<>(m_factory, m_v.minus(i_v.m_v));
    }

    public DifferentialVectorFunction<X> mul(long i_n) {
        return new DifferentialVectorFunction<>(m_factory, m_v.mul(i_n));
    }

    public DifferentialVectorFunction<X> mul(DifferentialFunction<X> i_v) {
        return new DifferentialVectorFunction<>(m_factory, m_v.mul(i_v));
    }

    public DifferentialVectorFunction<X> div(DifferentialFunction<X> i_v) {
        return new DifferentialVectorFunction<>(m_factory, m_v.div(i_v));
    }

}
