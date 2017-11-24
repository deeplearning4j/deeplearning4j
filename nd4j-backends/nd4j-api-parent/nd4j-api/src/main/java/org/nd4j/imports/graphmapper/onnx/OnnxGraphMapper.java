package org.nd4j.imports.graphmapper.onnx;

import com.google.common.primitives.Ints;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.BaseGraphMapper;
import org.nd4j.imports.graphmapper.ImportState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A mapper for onnx graphs to
 * {@link org.nd4j.autodiff.samediff.SameDiff} instances.
 *
 * @author Adam Gibson
 */
public class OnnxGraphMapper extends BaseGraphMapper<OnnxProto3.GraphProto, OnnxProto3.NodeProto, OnnxProto3.AttributeProto,  onnx.OnnxProto3.TypeProto.TensorTypeProto> {
    private static OnnxGraphMapper INSTANCE = new OnnxGraphMapper();


    public static OnnxGraphMapper getInstance() {
        return INSTANCE;
    }

    @Override
    public Map<String, Integer> verticesForGraph(OnnxProto3.GraphProto graph, SameDiff sameDiff) {
        //map the names of the ndoes while accumulating the vertex ids
        //for each variable
        val variablesForGraph = variablesForGraph(graph);
        val indexMap = new HashMap<String,Integer>();
        for(val entry : variablesForGraph.entrySet()) {
            val var = sameDiff.var(entry.getKey(),getNDArrayFromTensor(entry.getKey(), entry.getValue(), graph));
            indexMap.put(entry.getKey(),var.getVertexId()[0]);
        }

        return indexMap;
    }

    /**
     *
     * @param name the tensorflow or onnx name
     * @return
     */
    @Override
    public DifferentialFunction getMappedOp(String name) {
        return DifferentialFunctionClassHolder.getInstance().getOpWithOnnxName(name);
    }


    @Override
    public Map<String, Pair<int[], int[]>> inputsAndOutputsForGraph(OnnxProto3.GraphProto graph, Map<String, Integer> nodeNameToVertexId) {
        val ret = new HashMap<String, Pair<int[], int[]>>(graph.getNodeCount());
        for(val node : graph.getNodeList()) {
            val inputs = new int[node.getInputCount()];
            val outputs = new int[node.getOutputCount()];
            for(int i = 0; i < inputs.length; i++) {
                inputs[i] = nodeNameToVertexId.get(node.getInput(i));
            }

            for(int i = 0; i < outputs.length; i++) {
                outputs[i] = nodeNameToVertexId.get(node.getOutput(i));
            }

            ret.put(node.getName(),Pair.of(inputs,outputs));
        }

        return ret;
    }

    @Override
    public Map<String,  onnx.OnnxProto3.TypeProto.TensorTypeProto> variablesForGraph(OnnxProto3.GraphProto graphProto) {
        Map<String,  onnx.OnnxProto3.TypeProto.TensorTypeProto> ret = new HashMap<>();

        for(int i = 0; i < graphProto.getInputCount(); i++) {
            ret.put(graphProto.getInput(i).getName(),graphProto.getInput(i).getType().getTensorType());
        }

        for(int i = 0; i < graphProto.getOutputCount(); i++) {
            ret.put(graphProto.getOutput(i).getName(),graphProto.getOutput(i).getType().getTensorType());

        }


        return ret;
    }


    @Override
    public Message.Builder getNewGraphBuilder() {
        return OnnxProto3.GraphProto.newBuilder();
    }

    @Override
    public OnnxProto3.GraphProto parseGraphFrom(InputStream inputStream) throws IOException {
        return OnnxProto3.ModelProto.parseFrom(inputStream).getGraph();
    }

    @Override
    public void mapNodeType(OnnxProto3.NodeProto tfNode, ImportState<OnnxProto3.GraphProto,  onnx.OnnxProto3.TypeProto.TensorTypeProto> importState) {
        val differentialFunction = DifferentialFunctionClassHolder.getInstance().getOpWithOnnxName(tfNode.getOpType());
        if(differentialFunction == null) {
            throw new NoOpNameFoundException("No op name found " + tfNode.getName());
        }
        val diff = importState.getSameDiff();

        try {
            val newInstance = differentialFunction.getClass().newInstance();
            newInstance.initFromOnnx(tfNode,diff,getAttrMap(tfNode),importState.getGraph());
            val indices = importState.getVertexIdMap().get(tfNode.getName());
            val opStateEdge = getOpStateEdge(indices.getFirst(),indices.getSecond(),tfNode);
            newInstance.setVertexId(indices.getRight());
            newInstance.setSameDiff(importState.getSameDiff());
            /**
             * Need to f
             */
            diff.graph().addEdge(opStateEdge);
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }



    }



    @Override
    public DataBuffer.Type dataTypeForTensor( onnx.OnnxProto3.TypeProto.TensorTypeProto tensorProto) {
        switch (tensorProto.getElemType()) {
            case DOUBLE: return DataBuffer.Type.DOUBLE;
            case FLOAT: return DataBuffer.Type.FLOAT;
            case FLOAT16: return DataBuffer.Type.HALF;
            case INT32:
            case INT64: return DataBuffer.Type.INT;
            default: return DataBuffer.Type.UNKNOWN;
        }
    }



    @Override
    public String getAttrValueFromNode(OnnxProto3.NodeProto nodeProto, String key) {
        for(OnnxProto3.AttributeProto attributeProto : nodeProto.getAttributeList()) {
            if(attributeProto.getName().equals(key)) {
                return attributeProto.getS().toString();
            }
        }

        throw new ND4JIllegalStateException("No key found for " + key);
    }

    @Override
    public int[] getShapeFromAttribute(OnnxProto3.AttributeProto attributeProto) {
        return Ints.toArray(attributeProto.getT().getDimsList());
    }

    @Override
    public boolean isPlaceHolder(OnnxProto3.NodeProto nodeProto) {
        return false;
    }

    @Override
    public INDArray getNDArrayFromTensor(String tensorName, OnnxProto3.TypeProto.TensorTypeProto tensorProto, OnnxProto3.GraphProto graph) {
        DataBuffer.Type type = dataTypeForTensor(tensorProto);
        if(!tensorProto.isInitialized()) {
            throw new ND4JIllegalStateException("Unable to retrieve ndarray. Tensor was not initialized");
        }

        OnnxProto3.TensorProto tensor = null;
        for(int i = 0; i < graph.getInitializerCount(); i++) {
            val initializer = graph.getInitializer(i);
            if(initializer.getName().equals(tensorName)) {
                tensor = initializer;
                break;
            }
        }

        if(tensor == null)
            return null;

        ByteString bytes = tensor.getRawData();
        ByteBuffer byteBuffer = bytes.asReadOnlyByteBuffer().order(ByteOrder.nativeOrder());
        ByteBuffer directAlloc = ByteBuffer.allocateDirect(byteBuffer.capacity()).order(ByteOrder.nativeOrder());
        directAlloc.put(byteBuffer);
        directAlloc.rewind();
        int[] shape = getShapeFromTensor(tensorProto);
        DataBuffer buffer = Nd4j.createBuffer(directAlloc,type, ArrayUtil.prod(shape));
        INDArray arr = Nd4j.create(buffer).reshape(shape);
        return arr;
    }

    @Override
    public int[] getShapeFromTensor(onnx.OnnxProto3.TypeProto.TensorTypeProto tensorProto) {
        val ret = new int[Math.max(2,tensorProto.getShape().getDimCount())];
        int dimCount = tensorProto.getShape().getDimCount();
        if(dimCount >= 2)
            for(int i = 0; i < ret.length; i++) {
                ret[i] = (int) tensorProto.getShape().getDim(i).getDimValue();
            }
        else {
            ret[0] = 1;
            for(int i = 1; i < ret.length; i++) {
                ret[i] = (int) tensorProto.getShape().getDim(i - 1).getDimValue();
            }
        }


        return ret;
    }


    @Override
    public String getInputFromNode(OnnxProto3.NodeProto node, int index) {
        return node.getInput(index);
    }

    @Override
    public int numInputsFor(OnnxProto3.NodeProto nodeProto) {
        return nodeProto.getInputCount();
    }


    @Override
    public int[] getShapeFromAttr(OnnxProto3.AttributeProto attr) {
        return Ints.toArray(attr.getT().getDimsList());
    }

    @Override
    public Map<String, OnnxProto3.AttributeProto> getAttrMap(OnnxProto3.NodeProto nodeProto) {
        Map<String,OnnxProto3.AttributeProto> proto = new HashMap<>();
        for(int i = 0; i < nodeProto.getAttributeCount(); i++) {
            OnnxProto3.AttributeProto attributeProto = nodeProto.getAttribute(i);
            proto.put(attributeProto.getName(),attributeProto);
        }
        return proto;
    }

    @Override
    public String getName(OnnxProto3.NodeProto nodeProto) {
        return nodeProto.getName();
    }

    @Override
    public boolean alreadySeen(OnnxProto3.NodeProto nodeProto) {
        return false;
    }

    @Override
    public boolean isVariableNode(OnnxProto3.NodeProto nodeProto) {
        return nodeProto.getOpType().contains("Var");
    }

    @Override
    public boolean shouldSkip(OnnxProto3.NodeProto opType) {
        return false;
    }

    @Override
    public boolean hasShape(OnnxProto3.NodeProto nodeProto) {
        return false;
    }

    @Override
    public int[] getShape(OnnxProto3.NodeProto nodeProto) {
        return null;
    }

    @Override
    public INDArray getArrayFrom(OnnxProto3.NodeProto nodeProto, OnnxProto3.GraphProto graph) {

        return null;
    }

    @Override
    public String getOpType(OnnxProto3.NodeProto nodeProto) {
        return nodeProto.getOpType();
    }

    @Override
    public List<OnnxProto3.NodeProto> getNodeList(OnnxProto3.GraphProto graphProto) {
        return graphProto.getNodeList();
    }


}
