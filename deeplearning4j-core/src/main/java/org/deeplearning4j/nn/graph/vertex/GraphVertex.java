package org.deeplearning4j.nn.graph.vertex;

import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.nodes.GraphNode;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Arrays;

/** A graph vertex is a vertex in the computation graph. It may contain either a Layer, or a GraphNode
 * The purpose of  the GraphVertex class is as follows:
 * 1. To track the (local) network connection structure: i.e., it knows about the nodes on the input and output sides
 * 2. To store intermediate results (activations and epsilons)
 * 3. To allow forward pass and backward pass to be conducted, once the intermediate results are
 *
 */
@Data
public class GraphVertex implements Serializable {

    private ComputationGraph graph;

    private String vertexName;

    /** The index of this vertex */
    private int vertexIndex;

    /**A representation of the vertices that are inputs to this vertex (inputs during forward pass)
     * Specifically, if inputVertices[X].getVertexIndex() = Y, and inputVertices[X].getVertexEdgeNumber() = Z
     * then the Zth output of vertex Y is the Xth input to this vertex
     */
    private VertexIndices[] inputVertices;

    /**A representation of the vertices that this vertex is connected to (outputs duing forward pass)
     * Specifically, if outputVertices[X].getVertexIndex() = Y, and outputVertices[X].getVertexEdgeNumber() = Z
     * then the output of this vertex (there is only one output) is connected to the Zth input of vertex Y
     */
    private VertexIndices[] outputVertices;

    private Layer layer;
    private InputPreProcessor layerPreProcessor;
    private GraphNode node;

    private INDArray[] inputs;
    private INDArray[] epsilons;

    public GraphVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices,
                       Layer layer, InputPreProcessor inputPreProcessor ){
        this(graph, name, vertexIndex,inputVertices,outputVertices,layer,inputPreProcessor,null);
    }

    public GraphVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices, GraphNode graphNode){
        this(graph, name, vertexIndex,inputVertices,outputVertices,null,null,graphNode);
    }

    /** Create a network input vertex: */
    public GraphVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] outputVertices){
        this(graph, name, vertexIndex,null,outputVertices,null,null,null);
    }

    private GraphVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices,
                        Layer layer, InputPreProcessor layerPreProcessor, GraphNode graphNode){
        this.graph = graph;
        this.vertexName = name;
        this.vertexIndex = vertexIndex;
        this.inputVertices = inputVertices;
        this.outputVertices = outputVertices;
        this.layer = layer;
        this.layerPreProcessor = layerPreProcessor;
        this.node = graphNode;

        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
        this.epsilons = new INDArray[(outputVertices != null ? outputVertices.length : 0)];
    }

    public int getIndex(){
        return vertexIndex;
    }

    public int getNumInputArrays(){
        return (inputVertices == null ? 0 : inputVertices.length);
    }

    public int getNumOutputConnections(){
        return (outputVertices == null ? 0 : outputVertices.length);
    }

    /**A representation of the vertices that are inputs to this vertex (inputs duing forward pass)<br>
     * Specifically, if inputVertices[X].getVertexIndex() = Y, and inputVertices[X].getVertexEdgeNumber() = Z
     * then the Zth output of vertex Y is the Xth input to this vertex
     */
    public VertexIndices[] getInputVertices(){
        return inputVertices;
    }

    public void setInputVertices(VertexIndices[] inputVertices){
        this.inputVertices = inputVertices;
        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
    }

    /**A representation of the vertices that this vertex is connected to (outputs duing forward pass)
     * Specifically, if outputVertices[X].getVertexIndex() = Y, and outputVertices[X].getVertexEdgeNumber() = Z
     * then the Xth output of this vertex is connected to the Zth input of vertex Y
     */
    public VertexIndices[] getOutputVertices(){
        return outputVertices;
    }

    public void setOutputVertices(VertexIndices[] outputVertices){
        this.outputVertices = outputVertices;
        this.epsilons = new INDArray[(outputVertices != null ? outputVertices.length : 0)];
    }

    public boolean hasLayer(){
        return layer != null;
    }

    public boolean isInputVertex(){
        return layer == null && node == null;
    }

    public boolean isOutputVertex(){
        return (layer != null && layer instanceof BaseOutputLayer);
    }

    public Layer getLayer(){
        return layer;
    }

    public GraphNode getGraphNode(){
        return node;
    }

    public void setInput(int inputNumber, INDArray input){
        if(inputNumber >= getNumInputArrays()) throw new IllegalArgumentException("Invalid input number");
        inputs[inputNumber] = input;
    }

    public void setError(int errorNumber, INDArray error){
        if(errorNumber >= getNumOutputConnections() ){
            throw new IllegalArgumentException("Invalid error number: " + errorNumber
                    + ", numOutputEdges = " + outputVertices.length );
        }
        epsilons[errorNumber] = error;
    }

    public void clear(){
        for( int i=0; i<inputs.length; i++ ) inputs[i] = null;
        for( int i=0; i< epsilons.length; i++ ) epsilons[i] = null;
    }

    public boolean canDoForward(){
        for (INDArray input : inputs) if (input == null) return false;
        return true;
    }

    public boolean canDoBackward(){
        for (INDArray input : inputs) if (input == null) return false;
        return true;
    }

    public INDArray doForward(boolean training){
        if(!canDoForward()) throw new IllegalStateException("Cannot do forward pass: all inputs not set");

        if(layer != null){
            if(inputs.length > 1) throw new UnsupportedOperationException("Not implemented");   //TODO
            INDArray currInput = inputs[0];
            if(layerPreProcessor != null){
                currInput = layerPreProcessor.preProcess(currInput, graph.batchSize());
            }
            return layer.activate(currInput,training);
        } else {
            return node.forward(inputs);
        }
    }

    public Pair<Gradient,INDArray[]> doBackward(){
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: all epsilons not set");

        INDArray epsTotal = null;
        if(epsilons != null && epsilons.length == 1 ) epsTotal = epsilons[0];
        else if(epsilons != null && epsilons.length > 1 ){
            //TODO: check the math on this... I think it's correct though
            //This is the "output connected to multiple other layers" case
            epsTotal = epsilons[0].dup();
            for( int i=1; i<epsilons.length; i++ ){
                epsTotal.addi(epsilons[i]);
            }
        }

        if(layer != null){

            Pair<Gradient,INDArray> pair = layer.backpropGradient(epsTotal);    //epsTotal may be null for OutputLayers
            if(layerPreProcessor != null){
                INDArray eps = pair.getSecond();
                eps = layerPreProcessor.backprop(eps,graph.batchSize());
                pair.setSecond(eps);
            }

            //Layers always have single activations input -> always have single epsilon output during backprop
            return new Pair<>(pair.getFirst(), new INDArray[]{pair.getSecond()});
        } else {
            return new Pair<>(null,node.backward(epsTotal));    //No gradients for GraphNodes
        }
    }


    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("GraphVertex(id=").append(vertexIndex).append(",name=\"").append(vertexName)
                .append("\",inputs=").append(Arrays.toString(inputVertices)).append(",outputs=").append(Arrays.toString(outputVertices))
                .append(")");
        return sb.toString();
    }

}
