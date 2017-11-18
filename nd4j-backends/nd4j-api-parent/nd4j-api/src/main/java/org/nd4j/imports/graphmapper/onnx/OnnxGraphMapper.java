package org.nd4j.imports.graphmapper.onnx;

import com.google.common.primitives.Ints;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.opstate.OpStateEdge;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.BaseGraphMapper;
import org.nd4j.imports.graphmapper.ImportState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DefaultOpConverter;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A mapper for onnx graphs to {@link org.nd4j.autodiff.samediff.SameDiff} instances.
 *
 * @author Adam Gibson
 */
public class OnnxGraphMapper extends BaseGraphMapper<OnnxProto3.GraphProto, OnnxProto3.NodeProto, OnnxProto3.AttributeProto, OnnxProto3.TensorProto> {

    @Override
    public Map<String, OnnxProto3.TensorProto> variablesForGraph(OnnxProto3.GraphProto graphProto) {
        Map<String, OnnxProto3.TensorProto> ret = new HashMap<>();
        for(int i = 0; i < graphProto.getInitializerCount(); i++) {
            ret.put(graphProto.getInitializer(i).getName(),graphProto.getInitializer(i));
        }

        return ret;
    }

    @Override
    public Op.Type opTypeForNode(OnnxProto3.NodeProto nodeProto) {
        DifferentialFunction opWithOnnxName = DifferentialFunctionClassHolder.getInstance().getOpWithOnnxName(nodeProto.getOpType());
        if(opWithOnnxName == null)
            throw new NoOpNameFoundException("No onnx op found for " + nodeProto.getOpType());
        return opWithOnnxName.opType();
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
    public void mapNodeType(OnnxProto3.NodeProto tfNode, ImportState<OnnxProto3.GraphProto, OnnxProto3.TensorProto> importState) {
        int[] inputVertexIds = new int[tfNode.getInputCount()];
        int[] outputVertexIds = new int[tfNode.getOutputCount()];
        String[] vertexIdsForOpState = new String[tfNode.getInputCount() + tfNode.getOutputCount()];
        int vertexIdForOpStateIdx = 0;


        for(int i = 0; i < tfNode.getInputCount(); i++) {
            String input = tfNode.getInput(i);
            inputVertexIds[i] = Integer.parseInt(input);
            vertexIdsForOpState[vertexIdForOpStateIdx++] = input;
        }




        for(int i = 0; i < tfNode.getOutputCount(); i++) {
            String input = tfNode.getOutput(i);
            outputVertexIds[i] = Integer.parseInt(input);
            vertexIdsForOpState[vertexIdForOpStateIdx++] = input;

        }


        /**
         * Need to setup actual op here.
         * This includes mapping op attributes
         * from the protobuf to their DifferentialFunction counter parts in samediff.
         *
         * Note that this currently happens in tointermediaRepresentation
         */

        OpState opState = OpState.builder()
                .opType(opTypeForNode(tfNode))
                .vertexIds(vertexIdsForOpState)
                .opName(DifferentialFunctionClassHolder.getInstance().getOpWithOnnxName(tfNode.getOpType()).opName())
                .build();


        OpStateEdge opStateEdge = new OpStateEdge(inputVertexIds,outputVertexIds,opState,true);
        importState.getSameDiff().graph().addEdge(opStateEdge);

    }

    protected void addVarFromValueInfo(OnnxProto3.ValueInfoProto valueInfo, ImportState<OnnxProto3.GraphProto, OnnxProto3.TensorProto> importState, int i) {
        int[] shape = shapeFrom(valueInfo.getType().getTensorType().getShape().getDimList());
        OnnxProto3.TensorProto tensorProto = importState.getGraph().getInitializer(i);
        if(tensorProto != null) {
            importState.getSameDiff().var(String.valueOf(i),getNDArrayFromTensor(tensorProto));
        }
        else {
            importState.getSameDiff().var(String.valueOf(i),shape);
        }
    }


    @Override
    public DataBuffer.Type dataTypeForTensor(OnnxProto3.TensorProto tensorProto) {
        switch (tensorProto.getDataType()) {
            case DOUBLE: return DataBuffer.Type.DOUBLE;
            case FLOAT: return DataBuffer.Type.FLOAT;
            case FLOAT16: return DataBuffer.Type.HALF;
            case INT32:
            case INT64: return DataBuffer.Type.INT;
            default: return DataBuffer.Type.UNKNOWN;
        }
    }

    @Override
    public TOp asIntermediate(OnnxProto3.NodeProto nodeProto, TGraph intermediateGraph, Map<String, OnnxProto3.AttributeProto> attributes) {
        // first we try to use special converters
        DifferentialFunction converter = DifferentialFunctionClassHolder.getInstance().getInstance(nodeProto.getName().toLowerCase());
        if(converter == null)
            converter = DifferentialFunctionClassHolder.getInstance().getInstance(DefaultOpConverter.getInstance().opName());
        return converter.asIntermediateRepresentation(nodeProto, intermediateGraph, attributes);

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
    public INDArray getNDArrayFromTensor(OnnxProto3.TensorProto tensorProto) {
        DataBuffer.Type type = dataTypeForTensor(tensorProto);
        ByteString bytes = tensorProto.getRawData();
        ByteBuffer byteBuffer = bytes.asReadOnlyByteBuffer();
        ByteBuffer directAlloc = ByteBuffer.allocateDirect(byteBuffer.capacity()).order(ByteOrder.nativeOrder());
        directAlloc.put(byteBuffer);
        directAlloc.rewind();
        int[] shape = getShapeFromTensor(tensorProto);
        DataBuffer buffer = Nd4j.createBuffer(directAlloc,type, ArrayUtil.prod(shape));
        INDArray arr = Nd4j.create(buffer);
        return arr;
    }

    @Override
    public int[] getShapeFromTensor(OnnxProto3.TensorProto tensorProto) {
        return Ints.toArray(tensorProto.getDimsList());
    }

    @Override
    public OnnxProto3.TensorProto getTensorFrom(OnnxProto3.AttributeProto attributeProto) {
        return attributeProto.getT();
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
    public String valueKey() {
        return null;
    }

    @Override
    public String shapeKey() {
        return null;
    }

    @Override
    public String dTypeKey() {
        return null;
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
    public INDArray getArrayFrom(OnnxProto3.NodeProto nodeProto) {

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

    private int[] shapeFrom(List<OnnxProto3.TypeProto.TensorShapeProto.Dimension> dims) {
        List<Integer> size =  new ArrayList<>();
        if(dims.size() == 1)
            size.add(1);
        for(int i = 0; i < dims.size(); i++) {
            size.add((int) dims.get(i).getDimValue());
        }



        return Ints.toArray(size);

    }
}
