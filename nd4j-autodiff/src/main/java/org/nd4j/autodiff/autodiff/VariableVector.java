package org.nd4j.autodiff.autodiff;

import java.util.Collection;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;

public class VariableVector<X extends Field<X>> extends DifferentialVectorFunction<X> {

    public VariableVector(Graph<NDArrayInformation,OpState> graph,AbstractIdentityFactory<X> i_factory, Variable<X>... i_v) {
        super(graph,i_factory, i_v);
    }

    public VariableVector(Graph<NDArrayInformation,OpState> graph,AbstractIdentityFactory<X> i_factory, Collection<Variable<X>> i_v) {
        super(graph,i_factory, i_v);
    }

    public Variable<X> get(int i) {
        return (Variable<X>) m_v.get(i);
    }

    public void assign(DifferentialVectorFunction<X> i_v) {
        final int SIZE = size();
        if (SIZE != size()) {
            // throw Error
            return;
        }
        for (int i = SIZE - 1; i >= 0; i--) {
            get(i).set(i_v.get(i).getValue());
        }
    }
}
