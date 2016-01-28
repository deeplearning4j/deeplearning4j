package org.deeplearning4j.nn.graph.vertex;

import lombok.Data;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Arrays;

/** BaseGraphVertex defines a set of common functionality for GraphVertex instances.
 */
@Data
public abstract class BaseGraphVertex implements GraphVertex {

    protected ComputationGraph graph;

    protected String vertexName;

    /** The index of this vertex */
    protected int vertexIndex;

    /**A representation of the vertices that are inputs to this vertex (inputs during forward pass)
     * Specifically, if inputVertices[X].getVertexIndex() = Y, and inputVertices[X].getVertexEdgeNumber() = Z
     * then the Zth output of vertex Y is the Xth input to this vertex
     */
    protected VertexIndices[] inputVertices;

    /**A representation of the vertices that this vertex is connected to (outputs duing forward pass)
     * Specifically, if outputVertices[X].getVertexIndex() = Y, and outputVertices[X].getVertexEdgeNumber() = Z
     * then the output of this vertex (there is only one output) is connected to the Zth input of vertex Y
     */
    protected VertexIndices[] outputVertices;

    protected INDArray[] inputs;
    protected INDArray[] epsilons;

    protected BaseGraphVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices){
        this.graph = graph;
        this.vertexName = name;
        this.vertexIndex = vertexIndex;
        this.inputVertices = inputVertices;
        this.outputVertices = outputVertices;

        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
        this.epsilons = new INDArray[(outputVertices != null ? outputVertices.length : 0)];
    }

    @Override
    public String getVertexName(){
        return vertexName;
    }

    @Override
    public int getVertexIndex(){
        return vertexIndex;
    }

    @Override
    public int getNumInputArrays(){
        return (inputVertices == null ? 0 : inputVertices.length);
    }

    @Override
    public int getNumOutputConnections(){
        return (outputVertices == null ? 0 : outputVertices.length);
    }

    /**A representation of the vertices that are inputs to this vertex (inputs duing forward pass)<br>
     * Specifically, if inputVertices[X].getVertexIndex() = Y, and inputVertices[X].getVertexEdgeNumber() = Z
     * then the Zth output of vertex Y is the Xth input to this vertex
     */
    @Override
    public VertexIndices[] getInputVertices(){
        return inputVertices;
    }

    @Override
    public void setInputVertices(VertexIndices[] inputVertices){
        this.inputVertices = inputVertices;
        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
    }

    /**A representation of the vertices that this vertex is connected to (outputs duing forward pass)
     * Specifically, if outputVertices[X].getVertexIndex() = Y, and outputVertices[X].getVertexEdgeNumber() = Z
     * then the Xth output of this vertex is connected to the Zth input of vertex Y
     */
    @Override
    public VertexIndices[] getOutputVertices(){
        return outputVertices;
    }

    @Override
    public void setOutputVertices(VertexIndices[] outputVertices){
        this.outputVertices = outputVertices;
        this.epsilons = new INDArray[(outputVertices != null ? outputVertices.length : 0)];
    }

    @Override
    public boolean isInputVertex(){
        return false;
    }

    @Override
    public void setInput(int inputNumber, INDArray input){
        if(inputNumber >= getNumInputArrays()) throw new IllegalArgumentException("Invalid input number");
        inputs[inputNumber] = input;
    }

    @Override
    public void setError(int errorNumber, INDArray error){
        if(errorNumber >= getNumOutputConnections() ){
            throw new IllegalArgumentException("Invalid error number: " + errorNumber
                    + ", numOutputEdges = " + (outputVertices != null ? outputVertices.length : 0) );
        }
        epsilons[errorNumber] = error;
    }

    @Override
    public void clear(){
        for( int i=0; i<inputs.length; i++ ) inputs[i] = null;
        for( int i=0; i< epsilons.length; i++ ) epsilons[i] = null;
    }

    @Override
    public boolean canDoForward(){
        for (INDArray input : inputs) if (input == null) return false;
        return true;
    }

    @Override
    public boolean canDoBackward(){
        for (INDArray input : inputs) if (input == null) return false;
        return true;
    }

    @Override
    public INDArray[] getErrors(){
        return epsilons;
    }

    @Override
    public void setErrors(INDArray... errors){
        this.epsilons = errors;
    }

}
