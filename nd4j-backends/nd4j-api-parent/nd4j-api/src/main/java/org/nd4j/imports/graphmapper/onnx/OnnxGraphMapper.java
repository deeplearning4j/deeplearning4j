package org.nd4j.imports.graphmapper.onnx;

import com.google.common.primitives.Ints;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.BaseGraphMapper;
import org.nd4j.imports.graphmapper.ImportState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

/**
 * A mapper for onnx graphs to
 * {@link org.nd4j.autodiff.samediff.SameDiff} instances.
 *
 * @author Adam Gibson
 */
public class OnnxGraphMapper extends BaseGraphMapper<OnnxProto3.GraphProto, OnnxProto3.NodeProto, OnnxProto3.AttributeProto,  onnx.OnnxProto3.TypeProto.Tensor> {
    private static OnnxGraphMapper INSTANCE = new OnnxGraphMapper();


    public static OnnxGraphMapper getInstance() {
        return INSTANCE;
    }


    @Override
    public void dumpBinaryProtoAsText(InputStream inputFile, File outputFile) {
        try {
            OnnxProto3.ModelProto graphDef = OnnxProto3.ModelProto.parseFrom(inputFile);
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFile,true));
            for(OnnxProto3.NodeProto node : graphDef.getGraph().getNodeList()) {
                bufferedWriter.write(node.toString() + "\n");
            }

            bufferedWriter.flush();
            bufferedWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    @Override
    public String getTargetMappingForOp(DifferentialFunction function) {
        return null;
    }

    @Override
    public OnnxProto3.NodeProto getNodeWithNameFromGraph(OnnxProto3.GraphProto graph, String name) {
        for(int i = 0; i < graph.getNodeCount(); i++) {
            val node = graph.getNode(i);
            if(node.getName().equals(name))
                return node;
        }

        return null;
    }

    @Override
    public boolean isPlaceHolderNode(OnnxProto3.TypeProto.Tensor node) {
        return false;
    }

    @Override
    public void dumpBinaryProtoAsText(File inputFile, File outputFile) {
        try {
            OnnxProto3.ModelProto graphDef = OnnxProto3.ModelProto.parseFrom(new BufferedInputStream(new FileInputStream(inputFile)));
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFile,true));
            for(OnnxProto3.NodeProto node : graphDef.getGraph().getNodeList()) {
                bufferedWriter.write(node.toString());
            }

            bufferedWriter.flush();
            bufferedWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
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
    public Map<String,onnx.OnnxProto3.TypeProto.Tensor> variablesForGraph(OnnxProto3.GraphProto graphProto) {
        /**
         * Need to figure out why
         * gpu_0/conv1_1 isn't present in VGG
         */
        Map<String,onnx.OnnxProto3.TypeProto.Tensor> ret = new HashMap<>();
        for(int i = 0; i < graphProto.getInputCount(); i++) {
            ret.put(graphProto.getInput(i).getName(),graphProto.getInput(i).getType().getTensorType());
        }

        for(int i = 0; i < graphProto.getOutputCount(); i++) {
            ret.put(graphProto.getOutput(i).getName(),graphProto.getOutput(i).getType().getTensorType());
        }

        for(int i = 0; i < graphProto.getNodeCount(); i++) {
            val node = graphProto.getNode(i);
            val name = node.getName().isEmpty() ? String.valueOf(i) : node.getName();
            //add -1 as place holder value representing the shape needs to be filled in
            if(!ret.containsKey(name)) {
                addDummyTensor(name,ret);
            }

            for(int j = 0; j < node.getInputCount(); j++) {
                if(!ret.containsKey(node.getInput(j))) {
                    addDummyTensor(node.getInput(j),ret);
                }
            }


            for(int j = 0; j < node.getOutputCount(); j++) {
                if(!ret.containsKey(node.getOutput(j))) {
                    addDummyTensor(node.getOutput(j),ret);
                }
            }
        }

        return ret;
    }

    @Override
    public String translateToSameDiffName(String name, OnnxProto3.NodeProto node) {
        return null;
    }


    protected void addDummyTensor(String name, Map<String, OnnxProto3.TypeProto.Tensor> to) {
        OnnxProto3.TensorShapeProto.Dimension dim =  OnnxProto3.TensorShapeProto.Dimension.
                newBuilder()
                .setDimValue(-1)
                .build();
        OnnxProto3.TypeProto.Tensor typeProto = OnnxProto3.TypeProto.Tensor.newBuilder()
                .setShape(
                        OnnxProto3.TensorShapeProto.newBuilder()
                                .addDim(dim)
                                .addDim(dim).build())
                .build();
        to.put(name,typeProto);
    }

    @Override
    public Message.Builder getNewGraphBuilder() {
        return OnnxProto3.GraphProto.newBuilder();
    }

    @Override
    public OnnxProto3.GraphProto parseGraphFrom(byte[] inputStream) throws IOException {
        return OnnxProto3.ModelProto.parseFrom(inputStream).getGraph();
    }

    @Override
    public OnnxProto3.GraphProto parseGraphFrom(InputStream inputStream) throws IOException {
        return OnnxProto3.ModelProto.parseFrom(inputStream).getGraph();
    }

    @Override
    public void mapNodeType(OnnxProto3.NodeProto tfNode, ImportState<OnnxProto3.GraphProto,onnx.OnnxProto3.TypeProto.Tensor> importState) {
        val differentialFunction = DifferentialFunctionClassHolder.getInstance().getOpWithOnnxName(tfNode.getOpType());
        if(differentialFunction == null) {
            throw new NoOpNameFoundException("No op name found " + tfNode.getOpType());
        }

        val diff = importState.getSameDiff();
        val idx = importState.getGraph().getNodeList().indexOf(tfNode);
        val name = !tfNode.getName().isEmpty() ? tfNode.getName() : String.valueOf(idx);
        try {
            val newInstance = differentialFunction.getClass().newInstance();
            val args = new SDVariable[tfNode.getInputCount()];

            newInstance.setSameDiff(importState.getSameDiff());

            newInstance.initFromOnnx(tfNode,diff,getAttrMap(tfNode),importState.getGraph());

        }
        catch (Exception e) {
            e.printStackTrace();
        }



    }



    @Override
    public DataBuffer.Type dataTypeForTensor( onnx.OnnxProto3.TypeProto.Tensor tensorProto) {
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
    public boolean isPlaceHolder(OnnxProto3.TypeProto.Tensor nodeType) {
        return false;
    }


    @Override
    public INDArray getNDArrayFromTensor(String tensorName, OnnxProto3.TypeProto.Tensor tensorProto, OnnxProto3.GraphProto graph) {
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
    public int[] getShapeFromTensor(onnx.OnnxProto3.TypeProto.Tensor tensorProto) {
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
    public Set<String> opsToIgnore() {
        return Collections.emptySet();
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
