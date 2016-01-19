package org.deeplearning4j.nn.graph.nodes;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/** A GraphNode is a component of a ComputationGraph, that handles the interactions between layers
 *  */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = ElementWiseNode.class, name = "elementWiseNode"),
        @JsonSubTypes.Type(value = MergeNode.class, name = "mergeNode"),
        @JsonSubTypes.Type(value = SubsetNode.class, name = "subsetNode")
})
public interface GraphNode extends Cloneable, Serializable {

    INDArray forward(INDArray... activations);


    INDArray[] backward(INDArray epsilon);

    GraphNode clone();

}
