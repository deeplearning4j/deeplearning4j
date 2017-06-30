package org.nd4j.autodiff.functions;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.TensorGradGraph;
import org.nd4j.linalg.util.ArrayUtil;

import java.lang.reflect.InvocationTargetException;

@NoArgsConstructor
public abstract class AbstractBinaryFunction<X extends Field<X>> extends DifferentialFunction<X> {

    protected DifferentialFunction<X> m_x1;
    protected DifferentialFunction<X> m_x2;

    public AbstractBinaryFunction(TensorGradGraph graph,
                                  DifferentialFunction<X> i_v1,
                                  DifferentialFunction<X> i_v2) {
        super(graph,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            m_x1 = i_v1;
            m_x2 = i_v2;
            this.graph = graph;

            addEdges(graph,i_v1,i_v2,functionName());
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }
    }

    public AbstractBinaryFunction(TensorGradGraph graph) {
        this.graph = graph;
    }


    @Override
    public DifferentialFunction<X>[] args() {
        return new DifferentialFunction[] {larg(),rarg()};
    }

    public DifferentialFunction<X> larg() {
        if(m_x1 == this)
            return m_x1.dup();
        return m_x1;
    }


    public DifferentialFunction<X> rarg() {
        if(m_x2 == this)
            return m_x2.dup();
        return m_x2;
    }

    @Override
    public DifferentialFunction<X> dup() {
        try {
            return getClass().getConstructor(graph.getClass(),larg().getClass(),rarg().getClass()).newInstance(graph,larg(),rarg());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
