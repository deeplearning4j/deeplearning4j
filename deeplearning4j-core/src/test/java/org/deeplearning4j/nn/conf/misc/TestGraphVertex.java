package org.deeplearning4j.nn.conf.misc;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.graph.ComputationGraph;

@AllArgsConstructor @Data @NoArgsConstructor
public class TestGraphVertex extends GraphVertex {

    private int firstVal;
    private int secondVal;

    @Override
    public GraphVertex clone() {
        return new TestGraphVertex(firstVal,secondVal);
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        throw new UnsupportedOperationException();
    }
}
