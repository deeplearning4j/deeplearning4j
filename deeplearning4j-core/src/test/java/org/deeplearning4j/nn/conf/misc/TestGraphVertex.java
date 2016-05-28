package org.deeplearning4j.nn.conf.misc;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor @Data @NoArgsConstructor
public class TestGraphVertex extends GraphVertex {

    private int firstVal;
    private int secondVal;

    @Override
    public GraphVertex clone() {
        return new TestGraphVertex(firstVal,secondVal);
    }

    @Override
    public int numParams(boolean backprop) {
        return 0;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        throw new UnsupportedOperationException();
    }
}
