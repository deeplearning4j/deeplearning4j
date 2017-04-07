package org.nd4j.autodiff.autodiff;

import java.util.Collection;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;

public class ConstantVector<X extends Field<X>> extends DifferentialVectorFunction<X> {

    public ConstantVector(Graph<NDArrayInformation,OpState> graph,AbstractIdentityFactory<X> i_factory, Constant<X>... i_v) {
        super(graph,i_factory, i_v);
    }

    public ConstantVector(Graph<NDArrayInformation,OpState> graph,AbstractIdentityFactory<X> i_factory, Collection<Constant<X>> i_v) {
        super(graph,i_factory, i_v);
    }

    public Constant<X> get(int i) {
        return (Constant<X>) m_v.get(i);
    }

}
