package org.deeplearning4j.nn.conf.graph.rnn;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.graph.ComputationGraph;

/**DuplicateToTimeSeriesVertex is a vertex that goes from 2d activations to a 3d time series activations, by means of
 * duplication. That is, given a 2d input with shape [numExamples,nIn] duplicate each row to give output of
 * [numExamples,nIn,timeSeriesLength], where the activations are the same for all time steps.<br>
 * This method is used for example in sequence to sequence models.<br>
 * <b>Note</b>: The length of the output time series (number of time steps) is determined by means of referencing one of the
 * inputs in the ComputationGraph. That is: Because the length of the time series may differ at runtime, we generally want the number
 * of time steps to match some other input; here, we are specifying the length of the output time series to be the same as
 * one of the input time series<br>
 * @author Alex Black
 */
public class DuplicateToTimeSeriesVertex extends GraphVertex {

    private String inputName;

    public DuplicateToTimeSeriesVertex(String inputName){
        this.inputName = inputName;
    }

    @Override
    public GraphVertex clone() {
        return new DuplicateToTimeSeriesVertex(inputName);
    }

    @Override
    public boolean equals(Object o) {
        if(!(o instanceof DuplicateToTimeSeriesVertex)) return false;
        DuplicateToTimeSeriesVertex d = (DuplicateToTimeSeriesVertex)o;
        if(inputName == null && d.inputName != null || inputName != null && d.inputName == null) return false;
        return inputName == null || inputName.equals(d.inputName);
    }

    @Override
    public int hashCode() {
        return 534806565 ^ (inputName != null ? inputName.hashCode() : 0);
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx) {
        return new org.deeplearning4j.nn.graph.vertex.impl.rnn.DuplicateToTimeSeriesVertex(graph,name,idx,inputName);
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        if(vertexInputs.length != 1) throw new InvalidInputTypeException("Invalid input type: cannot duplicate more than 1 input");
        if(vertexInputs[0].getType() != InputType.Type.FF){
            throw new InvalidInputTypeException("Invalid input type: cannot duplicate to time series non feed forward input (got: " + vertexInputs[0] + ")");
        }

        return InputType.recurrent();
    }
}
