package org.deeplearning4j.nn.graph.vertex;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
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
public interface GraphVertex extends Serializable {

    String getVertexName();

    int getVertexIndex();

    int getNumInputArrays();

    int getNumOutputConnections();

    /**A representation of the vertices that are inputs to this vertex (inputs duing forward pass)<br>
     * Specifically, if inputVertices[X].getVertexIndex() = Y, and inputVertices[X].getVertexEdgeNumber() = Z
     * then the Zth output of vertex Y is the Xth input to this vertex
     */
    VertexIndices[] getInputVertices();

    void setInputVertices(VertexIndices[] inputVertices);

    /**A representation of the vertices that this vertex is connected to (outputs duing forward pass)
     * Specifically, if outputVertices[X].getVertexIndex() = Y, and outputVertices[X].getVertexEdgeNumber() = Z
     * then the Xth output of this vertex is connected to the Zth input of vertex Y
     */
    VertexIndices[] getOutputVertices();

    void setOutputVertices(VertexIndices[] outputVertices);

    boolean hasLayer();

    boolean isInputVertex();

    boolean isOutputVertex();

    Layer getLayer();

    void setInput(int inputNumber, INDArray input);

    void setError(int errorNumber, INDArray error);

    void clear();

    boolean canDoForward();

    boolean canDoBackward();

    INDArray doForward(boolean training);

    Pair<Gradient,INDArray[]> doBackward(boolean tbptt, int tbpttBackwardLength);

    INDArray[] getInputs();

    INDArray[] getErrors();

    void setInputs(INDArray... inputs);

    void setErrors(INDArray... errors);
}
