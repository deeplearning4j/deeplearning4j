package org.nd4j.autodiff.autodiff;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.UUID;


public abstract class AbstractUnaryFunction<X extends Field<X>> extends DifferentialFunction<X> {

    private DifferentialFunction<X> m_x;


    public AbstractUnaryFunction(Graph<NDArrayInformation,OpState> graph, DifferentialFunction<X> i_v) {
        super(graph);
        if (i_v != null) {
            m_x = i_v;
            addEdges(graph,m_x,functionName());
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }

    protected void addEdges(Graph<NDArrayInformation,OpState> graph, DifferentialFunction<X> i_v1,String opName) {
        if(i_v1.getValue() instanceof ArrayField) {
            ArrayField v1 = (ArrayField) i_v1.getValue();
            //result
            NDArrayVertex newVertex = new NDArrayVertex(graph.getVertices().size() ,
                    NDArrayInformation.builder()
                            .id(UUID.randomUUID().toString())
                            .shape(v1.getInput().getShape()).build());
            graph.addVertex(newVertex);
            graph.addEdge(v1.getVertex().getIdx(),newVertex.vertexID(),
                    OpState.builder()
                            .opType(OpState.OpType.TRANSFORM)
                            .opName(opName).id(UUID.randomUUID().toString())
                            .vertexIds(new String[]{String.valueOf(v1.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                            .n(ArrayUtil.prod(v1.getInput().getShape()))
                            .build(),true);

        }
    }


    public DifferentialFunction<X> arg() {
        return m_x;
    }
}
