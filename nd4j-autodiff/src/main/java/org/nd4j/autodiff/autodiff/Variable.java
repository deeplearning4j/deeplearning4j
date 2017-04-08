package org.nd4j.autodiff.autodiff;

import java.util.List;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;


public class Variable<X extends Field<X>> extends DifferentialFunction<X> {

    private X m_x;
    private AbstractIdentityFactory<X> m_factory;
    private String m_name;

    private PreEvaluator<X> preEvaluator;

    protected Variable(Graph<NDArrayInformation,OpState> graph,String i_name, X i_v, AbstractIdentityFactory<X> i_factory) {
        this(graph,i_name, i_v, i_factory, null);
    }

    protected Variable(Graph<NDArrayInformation,OpState> graph, String i_name, X i_v,
                       AbstractIdentityFactory<X> i_factory,
                       PreEvaluator<X> preEvaluator) {
       super(graph);
        this.preEvaluator = preEvaluator;
        setName(i_name);
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

    private void setName(String i_name) {
        if (i_name != null) {
            m_name = i_name;// new String(i_name);
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }
    }

    public String getName() {
        return m_name;
    }

    public void set(X i_v) {
        if (i_v != null) {
            m_x = i_v;
        } else {
            throw new IllegalArgumentException("Input not null value.");
        }
    }

    @Override
    public boolean isVariable() {
        return true;
    }

    @Override
    public X doGetValue() {
        if (preEvaluator != null) {
            preEvaluator.update(this);
        }
        return m_x;
    }

    @Override
    public double getReal() {
        if (preEvaluator != null) {
            preEvaluator.update(this);
        }
        return m_x.getReal();
    }

    @Override
    public Constant<X> diff(Variable<X> i_v) {
        return (this == i_v) ? new One<>(graph, m_factory) : new Zero<>(graph, m_factory);
    }

    @Override
    public String toString() {
        return getName() + ":" + getValue();
    }

    @Override
    public String doGetFormula(List<Variable<X>> variables) {
        variables.add(this);
        return getName();
    }

    @Override
    public DifferentialFunction<X> div(DifferentialFunction<X> i_v) {
        return (i_v == this) ? new One<>(graph, m_factory) : super.mul(i_v.inverse());
    }

}
