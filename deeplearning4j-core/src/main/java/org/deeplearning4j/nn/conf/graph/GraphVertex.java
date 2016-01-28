package org.deeplearning4j.nn.conf.graph;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.graph.ComputationGraph;

/** A GraphVertex is a vertex in the computation graph. It may contain Layer, or define some arbitrary forward/backward pass
 * behaviour based on the inputs
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = ElementWiseVertex.class, name = "ElementWiseVertex"),
        @JsonSubTypes.Type(value = MergeVertex.class, name = "MergeVertex"),
        @JsonSubTypes.Type(value = SubsetVertex.class, name = "SubsetVertex"),
        @JsonSubTypes.Type(value = LayerVertex.class, name = "LayerVertex"),
        @JsonSubTypes.Type(value = LastTimeStepVertex.class, name = "LastTimeStepVertex"),
        @JsonSubTypes.Type(value = DuplicateToTimeSeriesVertex.class, name = "DuplicateToTimeSeriesVertex")
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

    /** Determine the type of output for this GraphVertex, given the specified inputs. Given that a GraphVertex may do arbitrary
     * processing or modifications of the inputs, the output types can be quite different to the input type(s).<br>
     * This is generally used to determine when to add preprocessors, as well as the input sizes etc for layers
     * @param vertexInputs The inputs to this vertex
     * @return The type of output for this vertex
     * @throws InvalidInputTypeException If the input type is invalid for this type of GraphVertex
     */
    public abstract InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException;

}
