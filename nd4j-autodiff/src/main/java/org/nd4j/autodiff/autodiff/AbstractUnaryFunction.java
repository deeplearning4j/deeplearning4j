package org.nd4j.autodiff.autodiff;

import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;


public abstract class AbstractUnaryFunction<X extends Field<X>> extends DifferentialFunction<X> {

    private DifferentialFunction<X> m_x;


    public AbstractUnaryFunction(Graph<NDArrayInformation,OpState> graph, DifferentialFunction<X> i_v) {
        super(graph);
        if (i_v != null) {
            m_x = i_v;
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }

    @Override
    public X getValue() {
        graph.freeze();
        X ret = doGetValue();
        graph.unfreeze();
        return ret;
    }

    public abstract X doGetValue();

    public DifferentialFunction<X> arg() {
        return m_x;
    }
}
