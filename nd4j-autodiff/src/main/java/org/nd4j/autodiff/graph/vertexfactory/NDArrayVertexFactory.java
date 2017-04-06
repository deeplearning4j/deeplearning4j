package org.nd4j.autodiff.graph.vertexfactory;

import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;

/**
 * Created by agibsonccc on 4/6/17.
 */
public class NDArrayVertexFactory implements VertexFactory<NDArrayInformation> {

    /**
     * @param vertexIdx
     * @param args
     * @return
     */
    @Override
    public Vertex<NDArrayInformation> create(int vertexIdx, Object[] args) {
        return new NDArrayVertex(vertexIdx,(int[]) args[0]);
    }
}
