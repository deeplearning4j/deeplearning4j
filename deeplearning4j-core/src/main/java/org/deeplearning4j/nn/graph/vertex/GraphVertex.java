package org.deeplearning4j.nn.graph.vertex;

import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.nodes.GraphNode;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/** A graph vertex is a vertex in the computation graph. It may contain either a Layer, or a GraphNode
 * The purpose of  the GraphVertex class is as follows:
 * 1. To track the (local) network connection structure: i.e., it knows about the nodes on the input and output sides
 * 2. To store intermediate results (activations and epsilons)
 * 3. To allow forward pass and backward pass to be conducted, once the intermediate results are
 *
 */
@Data
public class GraphVertex {

    private String vertexName;

    /** The index of this vertex */
    private int vertexIndex;

    /** The indices of the vertices that are inputs to this vertex (inputs during forward pass, epsilons during backprop).
     *  Suppose inputIndices[x] = y. This means that: vertex y is the xth input to this vertex
     */
    private int[] inputIndices;

    /**For each of the inputs to this array, which of the outputs of the input vertices are used?
     * Suppose inputIndices[x] = y, and inputIndicesOutputNumbers[x] = z.
     * This means that the zth output of vertex y is connected to input x of this vertex
     * (need to know this for backprop)
     */
    private int[] inputIndicesOutputNumbers;

    /** The indices of the output from this vertex (vertices that this vertex is connected to for forward pass, or inputs/epsilons for backprop)
     * Suppose outputIndices[x] = y
     * This means that the xth output of this vertex is connected to vertex y
     */
    private int[] outputIndices;

    /** For each of the outputs from this vertex,
     * Suppose outputIndices[x] = y, and outputIndicesInputNumbers[x] = z.
     * This means that the xth output of this vertex is connected to the zth input of vertex y
     * (need to know this for forward pass
     */
    private int[] outputIndicesInputNumbers;

    private Layer layer;
    private GraphNode node;

    private INDArray[] inputs;
    private INDArray[] epsilons;

    public GraphVertex(String name, int vertexIndex, int[] inputIndices, int[] inputIndicesOutputNumbers[], int[] outputIndices, Layer layer){
        this(name, vertexIndex,inputIndices,outputIndices,layer,null);
    }

    public GraphVertex(String name, int vertexIndex, int[] inputIndices, int[] outputIndices, GraphNode graphNode){
        this(name, vertexIndex,inputIndices,outputIndices,null,graphNode);
    }

    /** Create a network input vertex: */
    public GraphVertex(String name, int vertexIndex, int[] outputIndices ){
        this(name, vertexIndex,null,outputIndices,null,null);
    }

    private GraphVertex(String name, int vertexIndex, int[] inputIndices, int[] outputIndices, Layer layer, GraphNode graphNode){
        this.vertexName = name;
        this.vertexIndex = vertexIndex;
        this.inputIndices = inputIndices;
        this.outputIndices = outputIndices;
        this.layer = layer;
        this.node = graphNode;
    }

    public int getIndex(){
        return vertexIndex;
    }

    public int getNumInputArrays(){
        return (inputIndices == null ? 0 : inputIndices.length);
    }

    public int getNumOutputArrays(){
        return (outputIndices == null ? 0 : outputIndices.length);
    }

    /** Index of the vertices that feed into this one */
    public int[] getInputVertexIndices(){
        return inputIndices;
    }

    /** For the vertices that feed into this one (according to {@link #getInputVertexIndices()},
     * which of the outputs of that vertex are used here?
     * For example, suppose the structure is such that A -> B, and A has 3 outputs
     * which of A's outputs is actually connected to B?
     */
    public int[] getInputVertexOutputNumbers(){

    }

    public int[] getOutputVertexIndices(){
        return outputIndices;
    }

    public boolean hasLayer(){
        return layer != null;
    }

    public boolean isInputVertex(){
        return layer == null && node == null;
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


    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("GraphVertex(id=").append(vertexIndex).append(",name=").append(vertexName)
                .append(",inputs=").append(Arrays.toString(inputIndices)).append(",outputs=").append(Arrays.toString(outputIndices))
                .append(")");
        return sb.toString();
    }

}
