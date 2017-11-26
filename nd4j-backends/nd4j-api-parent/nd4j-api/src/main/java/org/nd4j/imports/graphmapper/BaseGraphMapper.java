package org.nd4j.imports.graphmapper;

import com.google.protobuf.Message;
import com.google.protobuf.TextFormat;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.EdgeId;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Base implementation for importing a graph
 * @param <GRAPH_TYPE> the type of graph
 * @param <NODE_TYPE> the type of node
 * @param <ATTR_TYPE> the attribute type
 */
@Slf4j
public abstract class BaseGraphMapper<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE,TENSOR_TYPE> implements GraphMapper<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE,TENSOR_TYPE> {

    /**
     *
     * @param inputIds the input ids for the node
     *                  (based on the vertex ids in a {@link org.nd4j.autodiff.graph.Graph}
     * @param outputIds the output ids for the node
     *                  {based on the vertex ids in a {@link org.nd4j.autodiff.graph.Graph}}
     * @param node the node to create the edge from
     * @return
     */
    @Override
    public EdgeId getOpStateEdge(int[] inputIds, int[] outputIds, NODE_TYPE node) {
        EdgeId edgeId = new EdgeId(inputIds,outputIds, UUID.randomUUID().toString(),true);
        return edgeId;
    }


    @Override
    public Op.Type opTypeForNode(NODE_TYPE nodeDef) {
        DifferentialFunction opWithTensorflowName = getMappedOp(getOpType(nodeDef));
        if(opWithTensorflowName == null)
            throw new NoOpNameFoundException("No op found with name " + getOpType(nodeDef));
        Op.Type type = opWithTensorflowName.opType();
        return type;

    }

    /**
     *
     * @param graphFile
     * @return
     */
    @Override
    public  SameDiff importGraph(File graphFile) {
        GRAPH_TYPE def = null;
        try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis)) {
            def = parseGraphFrom(bis);
        } catch (Exception e) {
            try (FileInputStream fis2 = new FileInputStream(graphFile); BufferedInputStream bis2 = new BufferedInputStream(fis2); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
                Message.Builder builder = getNewGraphBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line);//.append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                def = (GRAPH_TYPE) builder.build();
            } catch (Exception e2) {
                e2.printStackTrace();
            }
        }

        if (def == null)
            throw new ND4JIllegalStateException("Unknown format: " + graphFile.getAbsolutePath());


        return importGraph(def);
    }

    @Override
    public Map<String, NODE_TYPE> nameIndexForGraph(GRAPH_TYPE graph) {
        List<NODE_TYPE> nodes = getNodeList(graph);
        Map<String,NODE_TYPE> ret = new HashMap<>();
        for(NODE_TYPE node : nodes) {
            ret.put(getName(node),node);
        }
        return ret;
    }

    /**
     * This method converts given TF
     * @param tfGraph
     * @return
     */
    @Override
    public SameDiff importGraph(GRAPH_TYPE tfGraph) {
        SameDiff diff = SameDiff.create();
        ImportState<GRAPH_TYPE,TENSOR_TYPE> importState = new ImportState<>();
        importState.setNodeCount(0);
        importState.setSameDiff(diff);
        importState.setGraph(tfGraph);
        val variablesForGraph = variablesForGraph(tfGraph);
        importState.setVariables(variablesForGraph);
        //map the names of the nodes while accumulating the vertex ids
        //for each variable
        val indexMap = new HashMap<String,Integer>();
        for(Map.Entry<String,TENSOR_TYPE> entry : variablesForGraph.entrySet()) {
            if(dataTypeForTensor(entry.getValue()) == DataBuffer.Type.UNKNOWN) {
                val var = importState.getSameDiff().var(entry.getKey(),null,0);
                indexMap.put(entry.getKey(),var.getVertexId()[0]);
                continue;
            }

            val arr = getNDArrayFromTensor(entry.getKey(), entry.getValue(), tfGraph);
            if(arr != null) {
                val var = importState.getSameDiff().var(entry.getKey(),arr);
                indexMap.put(entry.getKey(),var.getVertexId()[0]);
            }
            else if(getShapeFromTensor(entry.getValue()) == null) {
                val var = importState.getSameDiff().var(entry.getKey(),null,0);
                indexMap.put(entry.getKey(),var.getVertexId()[0]);
            }
            else {
                val var = importState.getSameDiff().var(entry.getKey(),getShapeFromTensor(entry.getValue()));
                indexMap.put(entry.getKey(),var.getVertexId()[0]);
            }

        }

        //handle mapping vertex ids properly
        val inputsAndOutputs = inputsAndOutputsForGraph(tfGraph,indexMap);
        importState.setVertexIdMap(inputsAndOutputs);


        val tfNodesList = getNodeList(tfGraph);
        for (NODE_TYPE tfNode : tfNodesList) {
            mapNodeType(tfNode,importState);
        }

        return diff;
    }




    @Override
    public boolean validTensorDataType(TENSOR_TYPE tensorType) {
        return dataTypeForTensor(tensorType) != DataBuffer.Type.UNKNOWN;
    }

}
