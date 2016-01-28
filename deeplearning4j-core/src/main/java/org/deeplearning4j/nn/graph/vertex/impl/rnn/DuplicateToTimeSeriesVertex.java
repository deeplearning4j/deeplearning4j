package org.deeplearning4j.nn.graph.vertex.impl.rnn;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
public class DuplicateToTimeSeriesVertex extends BaseGraphVertex {

    private String inputName;
    private int inputVertexIndex;

    public DuplicateToTimeSeriesVertex(ComputationGraph graph, String name, int vertexIndex, String inputVertexName){
        this(graph,name,vertexIndex,null,null,inputVertexName);
    }

    public DuplicateToTimeSeriesVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                                       VertexIndices[] outputVertices, String inputName) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.inputName = inputName;
        this.inputVertexIndex = graph.getConfiguration().getNetworkInputs().indexOf(inputName);
        if(inputVertexIndex == -1)  throw new IllegalArgumentException("Invalid input name: \"" + inputName + "\" not found in list "
                + "of network inputs (" + graph.getConfiguration().getNetworkInputs() + ")");
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public boolean isOutputVertex() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training) {

        //First: work out the time series length
        int tsLength = graph.getInput(inputVertexIndex).size(2);
        int[] outShape = new int[]{inputs[0].size(0),inputs[0].size(1),tsLength};

        INDArray out = Nd4j.create(outShape);
        for( int i=0; i<tsLength; i++ ){
            out.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(i)},inputs[0]);
        }
        return out;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        //Because we duplicated for each time step: simply need to sum along time for errors/epsilons
        return new Pair<>(null,new INDArray[]{epsilons[0].sum(2)});
    }

    @Override
    public String toString(){
        return "DuplicateToTimeSeriesVertex(inputName=" + inputName + ")";
    }
}
