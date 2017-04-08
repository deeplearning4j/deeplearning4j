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


public abstract class AbstractBinaryFunction<X extends Field<X>> extends DifferentialFunction<X> {

    private DifferentialFunction<X> m_x1;
    private DifferentialFunction<X> m_x2;
    protected Graph<NDArrayInformation,OpState> graph;

    public AbstractBinaryFunction(Graph<NDArrayInformation,OpState>
                                          graph,DifferentialFunction<X> i_v1,
                                  DifferentialFunction<X> i_v2) {
        super(graph);
        if (i_v1 != null && i_v2 != null) {
            m_x1 = i_v1;
            m_x2 = i_v2;
            this.graph = graph;

            addEdges(graph,i_v1,i_v2,functionName());
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }
    }




    protected void addEdges(Graph<NDArrayInformation,OpState> graph,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName) {
        if(i_v1.getValue() instanceof ArrayField) {
            ArrayField v1 = (ArrayField) i_v1.getValue();
            ArrayField v2 = (ArrayField) i_v2.getValue();
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
            graph.addEdge(v2.getVertex().getIdx(),newVertex.vertexID(),
                    OpState.builder()
                            .opType(OpState.OpType.TRANSFORM)
                            .opName(opName).id(UUID.randomUUID().toString())
                            .vertexIds(new String[]{String.valueOf(v2.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                            .n(ArrayUtil.prod(v1.getInput().getShape()))
                            .build(),true);



        }
    }



    public DifferentialFunction<X> larg() {
        return m_x1;
    }


    public DifferentialFunction<X> rarg() {
        return m_x2;
    }
}
