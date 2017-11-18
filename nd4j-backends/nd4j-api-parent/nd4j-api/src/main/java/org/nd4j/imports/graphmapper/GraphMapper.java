package org.nd4j.imports.graphmapper;

import com.google.protobuf.Message;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

/**
 * Map graph proto types to
 *
 * {@link SameDiff} instances
 * @param <GRAPH_TYPE> the proto type for the graph
 * @param <NODE_TYPE> the proto type for the node
 * @param <ATTR_TYPE> the proto type for the attribute
 * @param <TENSOR_TYPE> the proto type for the tensor
 *@author Adam Gibson
 */
public interface GraphMapper<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE,TENSOR_TYPE> {


    /**
     * Get the variables for the given graph
     * @param graphType the graph to get the variables for
     * @return a map of variable name to tensor
     */
    Map<String,TENSOR_TYPE> variablesForGraph(GRAPH_TYPE graphType);

    /**
     * Returns an op type for the given input node
     * @param nodeType the node to use
     * @return the optype for the given node
     */
    Op.Type opTypeForNode(NODE_TYPE nodeType);

    /**
     * Returns a graph builder for initial definition and parsing.
     * @return
     */
    Message.Builder getNewGraphBuilder();

    /**
     * Parse a graph from an input stream
     * @param inputStream the input stream to load from
     * @return
     */
     GRAPH_TYPE parseGraphFrom(InputStream inputStream) throws IOException;


    /**
     * Map a node in to the import state covering
     * the {@link SameDiff} instance
     * @param tfNode the node to map
     * @param importState the current import state
     */
    void mapNodeType(NODE_TYPE tfNode, ImportState<GRAPH_TYPE,TENSOR_TYPE> importState);

    /**
     * Map the graph type to an intermediate graph representation
     * {@link TGraph}
     * @param tfGraph the graph to map
     * @return the mapped intermediate graph
     */
    TGraph importIntermediate(GRAPH_TYPE tfGraph);


    /**
     * Map a graph to a {@link SameDiff}
     * instance
     * @param graphType the graph to map
     * @return the mapped samediff instance
     */
    SameDiff mapGraph(GRAPH_TYPE graphType);

    /**
     *
     * @param tensorType
     * @return
     */
    DataBuffer.Type dataTypeForTensor(TENSOR_TYPE tensorType);


    TOp asIntermediate(NODE_TYPE nodeType, TGraph intermediateGraph, Map<String, ATTR_TYPE> attributes);

    /**
     *
     * @param nodeType
     * @param key
     * @return
     */
    String  getAttrValueFromNode(NODE_TYPE nodeType,String key);


    /**
     *
     * @param attrType
     * @return
     */
    int[] getShapeFromAttribute(ATTR_TYPE attrType);

    /**
     * Returns true if the given node is a place holder type
     * (think a yet to be determined shape)_
     * @param nodeType
     * @return
     */
    boolean isPlaceHolder(NODE_TYPE nodeType);

    /**
     *
     * @param tensorType
     * @return
     */
    INDArray getNDArrayFromTensor(TENSOR_TYPE tensorType);


    /**
     * Get the shape for the given tensor type
     * @param tensorType
     * @return
     */
    int[] getShapeFromTensor(TENSOR_TYPE tensorType);

    /**
     * Get a tensor from the given attribute
     * @param attrType
     * @return
     */
    TENSOR_TYPE getTensorFrom(ATTR_TYPE attrType);


    /**
     * Get the input node for the given node
     * @param node the node
     * @param index hte index
     * @return
     */
    String getInputFromNode(NODE_TYPE node, int index);

    /**
     * Get the number of inputs for a node.
     * @param nodeType the node to get the number of inputs for
     * @return
     */
    int numInputsFor(NODE_TYPE nodeType);

    /**
     * Whether the data type for the tensor is valid
     * for creating an {@link INDArray}
     * @param tensorType the tensor proto to test
     * @return
     */
    boolean validTensorDataType(TENSOR_TYPE tensorType);

    /**
     * The attribute key for data type
     * @return
     */
    String valueKey();

    /**
     * The attribute key for data type
     * @return
     */
    String shapeKey();

    /**
     * The attribute key for data type
     * @return
     */
    String dTypeKey();

    /**
     * Get the shape of the attribute value
     * @param attr the attribute value
     * @return the shape of the attribute if any or null
     */
    int[] getShapeFromAttr(ATTR_TYPE attr);

    /**
     * Get the attribute
     * map for given node
     * @param nodeType the node
     * @return the attribute map for the attribute
     */
    Map<String,ATTR_TYPE> getAttrMap(NODE_TYPE nodeType);

    /**
     * Get the name of the node
     * @param nodeType the node
     *                 to get the name for
     * @return
     */
    String getName(NODE_TYPE nodeType);

    /**
     *
     * @param nodeType
     * @return
     */
    boolean alreadySeen(NODE_TYPE nodeType);

    /**
     *
     * @param nodeType
     * @return
     */
    boolean isVariableNode(NODE_TYPE nodeType);

    /**
     *
     *
     * @param opType
     * @return
     */
    boolean shouldSkip(NODE_TYPE opType);

    /**
     *
     * @param nodeType
     * @return
     */
    boolean hasShape(NODE_TYPE nodeType);

    /**
     *
     * @param nodeType
     * @return
     */
    int[] getShape(NODE_TYPE nodeType);

    /**
     *
     * @param nodeType
     * @return
     */
    INDArray getArrayFrom(NODE_TYPE nodeType);


    String getOpType(NODE_TYPE nodeType);

    /**
     *
     * @param graphType
     * @return
     */
    List<NODE_TYPE> getNodeList(GRAPH_TYPE graphType);

    SameDiff importGraph(File graphFile);

    /**
     * This method converts given TF
     * @param tfGraph
     * @return
     */
    SameDiff importGraph(GRAPH_TYPE tfGraph);

}
