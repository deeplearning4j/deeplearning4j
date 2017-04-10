package org.nd4j.autodiff;

import java.util.ArrayList;
import java.util.Collection;

public class BasicDenseVector<X extends Field<X>> implements CommutativeGroup<BasicDenseVector<X>> {

    protected ArrayList<X> m_v;

    public BasicDenseVector(X... i_v) {
        m_v = new ArrayList<X>(i_v.length);
        for (X element : i_v) {
            m_v.add(element);
        }
    }

    public BasicDenseVector(Collection<? extends X> i_v) {
        m_v = new ArrayList<X>(i_v.size());
        m_v.addAll(i_v);
    }

    public int size() {
        return m_v.size();
    }

    public X get(int i) {
        return m_v.get(i);
    }

    public X dot(BasicDenseVector<X> i_v) {
        int size = this.size();
        if (size != i_v.size() && size > 0) {
            return null;
        }
        X ret = get(0).mul(i_v.get(0));
        for (int i = 1; i < size; i++) {
            ret = ret.plus(get(i).mul(i_v.get(i)));
        }
        return ret;
    }

    public BasicDenseVector<X> negate() {
        int size = this.size();
        ArrayList<X> v = new ArrayList<X>(size);
        for (int i = 0; i < size; i++) {
            v.add(this.get(i).negate());
        }
        return new BasicDenseVector<X>(v);
    }

    public BasicDenseVector<X> plus(BasicDenseVector<X> i_v) {
        int size = this.size();
        if (size != i_v.size()) {
            return null;
        }
        ArrayList<X> v = new ArrayList<X>(size);
        for (int i = 0; i < size; i++) {
            v.add(this.get(i).plus(i_v.get(i)));
        }
        return new BasicDenseVector<X>(v);
    }

    public BasicDenseVector<X> minus(BasicDenseVector<X> i_v) {
        int size = this.size();
        if (size != i_v.size()) {
            return null;
        }
        ArrayList<X> v = new ArrayList<X>(size);
        for (int i = 0; i < size; i++) {
            v.add(this.get(i).minus(i_v.get(i)));
        }
        return new BasicDenseVector<X>(v);
    }

    public BasicDenseVector<X> mul(double i_n) {
        int size = this.size();
        ArrayList<X> v = new ArrayList<X>(size);
        for (int i = 0; i < size; i++) {
            v.add(this.get(i).mul(i_n));
        }
        return new BasicDenseVector<X>(v);
    }

    public BasicDenseVector<X> mul(X i_v) {
        int size = this.size();
        ArrayList<X> v = new ArrayList<X>(size);
        for (int i = 0; i < size; i++) {
            v.add(this.get(i).mul(i_v));
        }
        return new BasicDenseVector<X>(v);
    }

    public BasicDenseVector<X> div(X i_v) {
        int size = this.size();
        ArrayList<X> v = new ArrayList<X>(size);
        for (int i = 0; i < size; i++) {
            v.add(this.get(i).div(i_v));
        }
        return new BasicDenseVector<X>(v);
    }

}
