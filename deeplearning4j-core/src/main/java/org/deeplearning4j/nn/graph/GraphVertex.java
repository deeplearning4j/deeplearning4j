package org.deeplearning4j.nn.graph;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.nodes.GraphNode;
import org.nd4j.linalg.api.ndarray.INDArray;

/** A graph vertex is a vertex in the computation graph. It may contain either a Layer, or a GraphNode
 * The purpose of  the GraphVertex class is as follows:
 * 1. To track the (local) network connection structure: i.e., it knows about the nodes on the input and output sides
 * 2. To store intermediate results (activations and epsilons)
 * 3. To allow forward pass and backward pass to be conducted, once the intermediate results are
 *
 */
public class GraphVertex {

    /** The index of this vertex */
    private int vertexIndex;

    /** The indices of the inputs to this vertex (inputs during forward pass, epsilons during backprop) */
    private int[] inputIndices;

    /** The indices of the output from this vertex (vertices that this vertex is connected to for forward pass, or inputs/epsilons for backprop)
     */
    private int[] outputIndices;

    private Layer layer;
    private GraphNode node;

    private INDArray[] inputs;
    private INDArray[] epsilons;

    public GraphVertex(int vertexIndex, int[] inputIndices, int[] outputIndices, Layer layer){
        this(vertexIndex,inputIndices,outputIndices,layer,null);
    }

    public GraphVertex(int vertexIndex, int[] inputIndices, int[] outputIndices, GraphNode graphNode){
        this(vertexIndex,inputIndices,outputIndices,null,graphNode);
    }

    private GraphVertex(int vertexIndex, int[] inputIndices, int[] outputIndices, Layer layer, GraphNode graphNode){
        this.vertexIndex = vertexIndex;
        this.inputIndices = inputIndices;
        this.outputIndices = outputIndices;
        this.layer = layer;
        this.node = graphNode;
    }

    public int getNumInputArrays(){
        return inputIndices.length;
    }

    public int getNumOutputArrays(){
        return outputIndices.length;
    }

    public boolean hasLayer(){
        return layer != null;
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
        if(errorNumber >= getNumOutputArrays() ) throw new IllegalArgumentException("Invalid error number");
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

    public INDArray[] doForward(boolean training){
        if(!canDoForward()) throw new IllegalStateException("Cannot do forward pass: all inputs not set");

        if(layer != null){
            INDArray activations = layer.activate(inputs[0],training);
            INDArray[] ret = new INDArray[outputIndices.length];
            for( int i=0; i<ret.length; i++ ) ret[i] = activations; //TODO: when to duplicate?
            return ret;
        } else {
            return node.forward(inputs);
        }
    }

    public Pair<Gradient,INDArray[]> doBackward(boolean training){
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: all epsilons not set");

        if(layer != null){
            INDArray epsTotal;
            if(epsilons.length == 1 ) epsTotal = epsilons[0];
            else {
                //TODO: check the math on this... I think it's correct though?
                //This is the "output connected to multiple other layers" case
                epsTotal = epsilons[0].dup();
                for( int i=1; i<epsilons.length; i++ ){
                    epsTotal.addi(epsilons[i]);
                }
            }
            Pair<Gradient,INDArray> pair = layer.backpropGradient(epsTotal);

            //Layers always have single activations input -> always have single epsilon output during backprop
            return new Pair<Gradient,INDArray[]>(pair.getFirst(), new INDArray[]{pair.getSecond()});
        } else {
            return new Pair<>(null,node.backward(epsilons));    //No gradients for GraphNodes
        }
    }



}
