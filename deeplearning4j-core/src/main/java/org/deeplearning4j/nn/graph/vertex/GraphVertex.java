package org.deeplearning4j.nn.graph.vertex;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/** A GraphVertex is a vertex in the computation graph. It may contain Layer, or define some arbitrary forward/backward pass
 * behaviour based on the inputs.<br>
 * The purposes of GraphVertex instances are as follows:
 * 1. To track the (local) network connection structure: i.e., a GraphVertex knows about the vertices on the input and output sides
 * 2. To store intermediate results (activations and epsilons)
 * 3. To allow forward pass and backward pass to be conducted, once the intermediate results are set
 * @author Alex Black
 */
public interface GraphVertex extends Serializable {

    /**Get the name/label of the GraphVertex
     */
    String getVertexName();

    /** Get the index of the GraphVertex */
    int getVertexIndex();

    /** Get the number of input arrays. For example, a Layer may have only one input array, but in general a GraphVertex
     * may have an arbtrary (>=1) number of input arrays (for example, from multiple other layers)
     */
    int getNumInputArrays();

    /** Get the number of outgoing connections from this GraphVertex. A GraphVertex may only have a single output (for
     * example, the activations out of a layer), but this output may be used as the input to an arbitrary number of other
     * GraphVertex instances. This method returns the number of GraphVertex instances the output of this GraphVertex is input for.
     */
    int getNumOutputConnections();

    /**A representation of the vertices that are inputs to this vertex (inputs duing forward pass)<br>
     * Specifically, if inputVertices[X].getVertexIndex() = Y, and inputVertices[X].getVertexEdgeNumber() = Z
     * then the Zth output connection (see {@link #getNumOutputConnections()} of vertex Y is the Xth input to this vertex
     */
    VertexIndices[] getInputVertices();

    /** Sets the input vertices.
     * @see #getInputVertices()
     */
    void setInputVertices(VertexIndices[] inputVertices);

    /**A representation of the vertices that this vertex is connected to (outputs duing forward pass)
     * Specifically, if outputVertices[X].getVertexIndex() = Y, and outputVertices[X].getVertexEdgeNumber() = Z
     * then the Xth output of this vertex is connected to the Zth input of vertex Y
     */
    VertexIndices[] getOutputVertices();

    /** set the output vertices.
     * @see #getOutputVertices()
     */
    void setOutputVertices(VertexIndices[] outputVertices);

    /** whether the GraphVertex contains a {@link Layer} object or not */
    boolean hasLayer();

    /** Whether the GraphVertex is an input vertex */
    boolean isInputVertex();

    /** Whether the GraphVertexis an output vertex */
    boolean isOutputVertex();

    /** Get the Layer (if any). Returns null if {@link #hasLayer()} == false */
    Layer getLayer();

    /** Set the input activations.
     *
     * @param inputNumber Must be in range 0 to {@link #getNumInputArrays()}-1
     * @param input The input array
     */
    void setInput(int inputNumber, INDArray input);

    /** Set the errors (epsilons) for this GraphVertex */
    void setError(int errorNumber, INDArray error);

    /** Clear the internal state (if any) of the GraphVertex. For example, any stored inputs/errors */
    void clear();

    /** Whether the GraphVertex can do forward pass. Typically, this is just whether all inputs are set. */
    boolean canDoForward();

    /** Whether the GraphVertex can do backward pass. Typically, this is just whether all errors/epsilons are set */
    boolean canDoBackward();

    /** Do forward pass using the stored inputs
     * @param training if true: forward pass at training time. If false: forward pass at test time
     * @return The output (for example, activations) of the GraphVertex
     */
    INDArray doForward(boolean training);

    /** Do backward pass
     * @param tbptt If true: do backprop using truncated BPTT
     * @return The gradients (may be null), and the errors/epsilons for all inputs to this GraphVertex
     */
    Pair<Gradient,INDArray[]> doBackward(boolean tbptt);

    /** Get the array of inputs previously set for this GraphVertex */
    INDArray[] getInputs();

    /** Get the array of errors previously set for this GraphVertex */
    INDArray[] getErrors();

    /** Set all inputs for this GraphVertex
     * @see #setInput(int, INDArray)
     */
    void setInputs(INDArray... inputs);

    /** Set all errors/epsilons for this GraphVertex
     * @see #setError(int, INDArray)
     */
    void setErrors(INDArray... errors);
}
