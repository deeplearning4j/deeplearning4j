/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.imports.graphmapper.onnx;

import com.github.os72.protobuf351.ByteString;
import com.github.os72.protobuf351.Message;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
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



    /**
     * Init a function's attributes
     * @param mappedTfName the onnx name to pick (sometimes ops have multiple names
     * @param on the function to map
     * @param attributesForNode the attributes for the node
     * @param node
     * @param graph
     */
    public void initFunctionFromProperties(String mappedTfName, DifferentialFunction on, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.NodeProto node, OnnxProto3.GraphProto graph) {
        val properties = on.mappingsForFunction();
        val tfProperties = properties.get(mappedTfName);
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(on);
        val attributeAdapters = on.attributeAdaptersForFunction();
        for(val entry : tfProperties.entrySet()) {
            val tfAttrName = entry.getValue().getTfAttrName();
            val currentField = fields.get(entry.getKey());

            AttributeAdapter adapter = null;
            if(tfAttrName != null) {
                if(currentField == null) {
                    continue;
                }
                if(attributeAdapters != null && !attributeAdapters.isEmpty()) {
                    val mappers = attributeAdapters.get(on.tensorflowName());
                    val adapterFor = mappers.get(entry.getKey());
                    adapter = adapterFor;
                }


                if(attributesForNode.containsKey(tfAttrName)) {
                    val attr = attributesForNode.get(tfAttrName);
                    switch (attr.getType()) {
                        case STRING:
                            val setString = attr.getS().toStringUtf8();
                            if(adapter != null) {
                                adapter.mapAttributeFor(setString,currentField,on);
                            }
                            else
                                on.setValueFor(currentField,setString);
                            break;
                        case INT:
                            val setInt = (int) attr.getI();
                            if(adapter != null) {
                                adapter.mapAttributeFor(setInt,currentField,on);
                            }
                            else
                                on.setValueFor(currentField,setInt);
                            break;
                        case INTS:
                            val setList = attr.getIntsList();
                            if(!setList.isEmpty()) {
                                val intList = Ints.toArray(setList);
                                if(adapter != null) {
                                    adapter.mapAttributeFor(intList,currentField,on);
                                }
                                else
                                    on.setValueFor(currentField,intList);
                            }
                            break;
                        case FLOATS:
                            val floatsList = attr.getFloatsList();
                            if(!floatsList.isEmpty()) {
                                val floats = Floats.toArray(floatsList);
                                if(adapter != null) {
                                    adapter.mapAttributeFor(floats,currentField,on);
                                }

                                else
                                    on.setValueFor(currentField,floats);
                                break;
                            }
                            break;
                        case TENSOR:
                            val tensorToGet = mapTensorProto(attr.getT());
                            if(adapter != null) {
                                adapter.mapAttributeFor(tensorToGet,currentField,on);
                            }
                            else
                                on.setValueFor(currentField,tensorToGet);
                            break;

                    }
                }
            }


        }
    }

    @Override
    public boolean isOpIgnoreException(OnnxProto3.NodeProto node) {
        return false;
    }

    @Override
    public String getTargetMappingForOp(DifferentialFunction function, OnnxProto3.NodeProto node) {
        return function.opName();
    }


    @Override
    public void mapProperty(String name, DifferentialFunction on, OnnxProto3.NodeProto node, OnnxProto3.GraphProto graph, SameDiff sameDiff, Map<String, Map<String, PropertyMapping>> propertyMappingsForFunction) {
        val mapping = propertyMappingsForFunction.get(name).get(getTargetMappingForOp(on, node));
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(on);
        /**
         * Map  ints and the like. Need to figure out how attribute mapping should work.
         *
         *
         */

        val propsForFunction = on.propertiesForFunction();

        if(mapping.getTfAttrName() == null) {
            int tfMappingIdx = mapping.getTfInputPosition();
            if(tfMappingIdx < 0)
                tfMappingIdx += node.getInputCount();

            val input = node.getInput(tfMappingIdx);
            val inputNode = getInstance().getNodeWithNameFromGraph(graph,input);
            INDArray arr = sameDiff.getArrForVarName(input);
            val field = fields.get(mapping.getPropertyNames()[0]);
            val type = field.getType();
            if(type.equals(int[].class)) {
                try {
                    field.set(arr.data().asInt(),on);
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
            else if(type.equals(int.class) || type.equals(long.class) || type.equals(Long.class) || type.equals(Integer.class)) {
                try {
                    field.set(arr.getInt(0),on);
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
            else if(type.equals(float.class) || type.equals(double.class) || type.equals(Float.class) || type.equals(Double.class)) {
                try {
                    field.set(arr.getDouble(0),on);
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }



            /**
             * Figure out whether it's an int array
             * or a double array, or maybe a scalar.
             */

        }
        else {
            val tfMappingAttrName = mapping.getOnnxAttrName();
            val attr = getAttrMap(node).get(tfMappingAttrName);
            val type = attr.getType();
            val field = fields.get(mapping.getPropertyNames()[0]);

            Object valueToSet = null;
            switch(type) {
                case INT:
                    valueToSet = attr.getI();
                    break;
                case FLOAT:
                    valueToSet = attr.getF();
                    break;
                case STRING:
                    valueToSet = attr.getF();
                    break;

            }

            try {
                field.set(valueToSet,on);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }

        }
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
            importState.getSameDiff().putFunctionForId(newInstance.getOwnName(),newInstance);
            //ensure we can track node name to function instance later.
            diff.setBaseNameForFunctionInstanceId(tfNode.getName(),newInstance);
            diff.addVarNameForImport(tfNode.getName());
        }
        catch (Exception e) {
            e.printStackTrace();
        }



    }



    @Override
    public DataBuffer.Type dataTypeForTensor( onnx.OnnxProto3.TypeProto.Tensor tensorProto) {
       return nd4jTypeFromOnnxType(tensorProto.getElemType());
    }

    @Override
    public boolean unknownTypeNodeImportable(OnnxProto3.TypeProto.Tensor tensor) {
        return false;
    }


    /**
     * Convert an onnx type to the proper nd4j type
     * @param dataType the data type to convert
     * @return the nd4j type for the onnx type
     */
    public DataBuffer.Type nd4jTypeFromOnnxType(OnnxProto3.TensorProto.DataType dataType) {
        switch (dataType) {
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
    public long[] getShapeFromAttribute(OnnxProto3.AttributeProto attributeProto) {
        return Longs.toArray(attributeProto.getT().getDimsList());
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
        long[] shape = getShapeFromTensor(tensorProto);
        DataBuffer buffer = Nd4j.createBuffer(directAlloc,type, ArrayUtil.prod(shape));
        INDArray arr = Nd4j.create(buffer).reshape(shape);
        return arr;
    }

    public INDArray mapTensorProto(OnnxProto3.TensorProto tensor) {
        if(tensor == null)
            return null;


        DataBuffer.Type type = nd4jTypeFromOnnxType(tensor.getDataType());

        ByteString bytes = tensor.getRawData();
        ByteBuffer byteBuffer = bytes.asReadOnlyByteBuffer().order(ByteOrder.nativeOrder());
        ByteBuffer directAlloc = ByteBuffer.allocateDirect(byteBuffer.capacity()).order(ByteOrder.nativeOrder());
        directAlloc.put(byteBuffer);
        directAlloc.rewind();
        long[] shape = getShapeFromTensor(tensor);
        DataBuffer buffer = Nd4j.createBuffer(directAlloc,type, ArrayUtil.prod(shape));
        INDArray arr = Nd4j.create(buffer).reshape(shape);
        return arr;
    }

    @Override
    public long[] getShapeFromTensor(onnx.OnnxProto3.TypeProto.Tensor tensorProto) {
        val ret = new long[Math.max(2,tensorProto.getShape().getDimCount())];
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


    /**
     * Get the shape from a tensor proto.
     * Note that this is different from {@link #getShapeFromTensor(OnnxProto3.TensorProto)}
     * @param tensorProto the tensor to get the shape from
     * @return
     */
    public long[] getShapeFromTensor(OnnxProto3.TensorProto tensorProto) {
        val ret = new long[Math.max(2,tensorProto.getDimsCount())];
        int dimCount = tensorProto.getDimsCount();
        if(dimCount >= 2)
            for(int i = 0; i < ret.length; i++) {
                ret[i] = (int) tensorProto.getDims(i);
            }
        else {
            ret[0] = 1;
            for(int i = 1; i < ret.length; i++) {
                ret[i] = (int) tensorProto.getDims(i - 1);
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
    public long[] getShapeFromAttr(OnnxProto3.AttributeProto attr) {
        return Longs.toArray(attr.getT().getDimsList());
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
    public long[] getShape(OnnxProto3.NodeProto nodeProto) {
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
