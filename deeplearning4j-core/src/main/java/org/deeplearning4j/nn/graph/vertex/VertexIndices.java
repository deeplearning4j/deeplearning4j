package org.deeplearning4j.nn.graph.vertex;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;

import java.io.Serializable;

@AllArgsConstructor
@EqualsAndHashCode
public class VertexIndices implements Serializable {

    private final int vertexIndex;
    private final int vertexEdgeNumber;


    /**Index of the vertex */
    public int getVertexIndex() {
        return this.vertexIndex;
    }

    /** The edge number. Represents the index of the output of the vertex index, OR the index of the
     * input to the vertex, depending on the context
     */
    public int getVertexEdgeNumber() {
        return this.vertexEdgeNumber;
    }

    public String toString() {
        return "VertexIndices(vertexIndex=" + this.vertexIndex + ", vertexEdgeNumber=" + this.vertexEdgeNumber + ")";
    }
}
