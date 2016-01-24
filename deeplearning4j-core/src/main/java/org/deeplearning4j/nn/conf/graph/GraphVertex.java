package org.deeplearning4j.nn.conf.graph;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import org.deeplearning4j.nn.graph.ComputationGraph;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = ElementWiseVertex.class, name = "ElementWiseVertex"),
        @JsonSubTypes.Type(value = MergeVertex.class, name = "MergeVertex"),
        @JsonSubTypes.Type(value = SubsetVertex.class, name = "SubsetVertex")
})
public abstract class GraphVertex implements Cloneable {

    @Override
    public abstract GraphVertex clone();

    @Override
    public abstract boolean equals(Object o);

    @Override
    public abstract int hashCode();

    /** Create a {@link org.deeplearning4j.nn.graph.vertex.GraphVertex} instance, for the given computation graph,
     * given the configuration instance.
     * @param graph The computation graph that this GraphVertex is to be part of
     * @param name The name of the GraphVertex object
     * @param idx The index of the GraphVertex
     * @return The implementation GraphVertex object (i.e., implementation, no the configuration)
     */
    public abstract org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx);

}
