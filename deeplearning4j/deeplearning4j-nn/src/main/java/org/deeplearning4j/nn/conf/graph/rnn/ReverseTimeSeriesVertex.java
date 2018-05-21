package org.deeplearning4j.nn.conf.graph.rnn;

import lombok.Data;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * ReverseTimeSeriesVertex is used in recurrent neural networks to revert the order of time series.
 * As a result, the last time step is moved to the beginning of the time series and the first time step
 * is moved to the end. This allows recurrent layers to backward process time series.<p>
 *
 * <b>Masks</b>: The input might be masked (to allow for varying time series lengths in one minibatch). In this case the
 * present input (mask array = 1) will be reverted in place and the padding (mask array = 0) will be left untouched at
 * the same place. For a time series of length n, this would normally mean, that the first n time steps are reverted and
 * the following padding is left untouched, but more complex masks are supported (e.g. [1, 0, 1, 0, ...].<br>
 * <b>Note</b>: In order to use mask arrays, the {@link #ReverseTimeSeriesVertex(String) constructor} must be called with
 * the name of an network input. The mask of this input is then used in this vertex, too.
 *
 * @author Klaus Broelemann (SCHUFA Holding AG)
 */
@Data
public class ReverseTimeSeriesVertex extends GraphVertex {
    private final String maskArrayInputName;

    /**
     * Creates a new ReverseTimeSeriesVertex that doesn't pay attention to masks
     */
    public ReverseTimeSeriesVertex() {
        this(null);
    }

    /**
     * Creates a new ReverseTimeSeriesVertex that uses the mask array of a given input
     * @param maskArrayInputName The name of the input that holds the mask.
     */
    public ReverseTimeSeriesVertex(String maskArrayInputName) {
        this.maskArrayInputName = maskArrayInputName;
    }

    public ReverseTimeSeriesVertex clone() {
        return new ReverseTimeSeriesVertex(maskArrayInputName);
    }

    public boolean equals(Object o) {
        if (!(o instanceof ReverseTimeSeriesVertex))
            return false;
        ReverseTimeSeriesVertex rsgv = (ReverseTimeSeriesVertex) o;
        if (maskArrayInputName == null && rsgv.maskArrayInputName != null
                || maskArrayInputName != null && rsgv.maskArrayInputName == null)
            return false;
        return maskArrayInputName == null || maskArrayInputName.equals(rsgv.maskArrayInputName);
    }

    @Override
    public int hashCode() {
        return maskArrayInputName != null ? maskArrayInputName.hashCode() : 0;
    }

    public int numParams(boolean backprop) {
        return 0;
    }

    public int minVertexInputs() {
        return 1;
    }

    public int maxVertexInputs() {
        return 1;
    }

    public org.deeplearning4j.nn.graph.vertex.impl.rnn.ReverseTimeSeriesVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
        return new org.deeplearning4j.nn.graph.vertex.impl.rnn.ReverseTimeSeriesVertex(graph, name, idx, maskArrayInputName);
    }

    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1)
            throw new InvalidInputTypeException("Invalid input type: cannot revert more than 1 input");
        if (vertexInputs[0].getType() != InputType.Type.RNN) {
            throw new InvalidInputTypeException(
                    "Invalid input type: cannot revert non RNN input (got: " + vertexInputs[0] + ")");
        }

        return vertexInputs[0];
    }

    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //No additional working memory (beyond activations/epsilons)
        return new LayerMemoryReport.Builder(null, getClass(), inputTypes[0], getOutputType(-1, inputTypes))
                .standardMemory(0, 0)
                .workingMemory(0, 0, 0, 0)
                .cacheMemory(0, 0)
                .build();
    }

    public String toString() {
        final String paramStr = (maskArrayInputName == null) ? "" : "inputName=" + maskArrayInputName;
        return "ReverseTimeSeriesVertex(" + paramStr + ")";
    }
}
