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

package org.nd4j.imports.graphmapper.tf;

import com.github.os72.protobuf351.Message;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.BaseGraphMapper;
import org.nd4j.imports.graphmapper.ImportState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.IfImportState;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.*;

import java.io.*;
import java.nio.ByteOrder;
import java.util.*;

/**
 * Map tensorflow graph protos
 * to the intermediate representation
 * for samediff.
 *
 * @author Adam Gibson
 */
@Slf4j
public class TFGraphMapper extends BaseGraphMapper<GraphDef,NodeDef,AttrValue,NodeDef> {
    private Set<String> seenNodes = new LinkedHashSet<>();
    public final static String VALUE_ATTR_KEY = "value";
    public final static String SHAPE_KEY = "shape";
    private static TFGraphMapper MAPPER_INSTANCE = new TFGraphMapper();
    private Set<String> graphMapper = new HashSet<String>(){{
        //While and If
        //While -> Enter
        /**
         * Need to work on coping with variables
         * that are marked as "shouldSkip"
         *
         * Possibly consider replacing should skip
         * with a special handler interface. Something like
         *
         * public interface ImportOpHandler
         */
        add("LoopCond");
        /**
         * We should skip this for the sake of while..but not if.
         * Need to be a bit more flexible here.
         */
        add("Merge");
        add("Exit");
        add("NextIteration");
        add("NoOp");
        add("Switch");
    }};
    //singleton
    private TFGraphMapper() {}

    /**
     * Singleton. Get the needed instance.
     * @return
     */
    public static TFGraphMapper getInstance() {
        return MAPPER_INSTANCE;
    }

    @Override
    public void dumpBinaryProtoAsText(InputStream inputFile, File outputFile) {
        try {
            GraphDef graphDef = GraphDef.parseFrom(inputFile);
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFile,true));
            for(NodeDef node : graphDef.getNodeList()) {
                bufferedWriter.write(node.toString());
            }

            bufferedWriter.flush();
            bufferedWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public boolean isOpIgnoreException(NodeDef node) {
        //if statements should not be ignored
/*
        if(node.getOp().equals("Merge")) {
            boolean ret = false;
            for(int i = 0; i < node.getInputCount(); i++) {
                //while loop
                ret = ret || !node.getInput(i).endsWith("/Enter") || !node.getInput(i).endsWith("/NextIteration");

            }

            return ret;
        }

        else if(node.getOp().equals("Switch")) {
            boolean ret = false;
            for(int i = 0; i < node.getInputCount(); i++) {
                //while loop
                ret = ret || !node.getInput(i).endsWith("/Merge") || !node.getInput(i).endsWith("/LoopCond");

            }

            return ret;
        }
*/
        return true;
    }

    @Override
    public String getTargetMappingForOp(DifferentialFunction function, NodeDef node) {
        return function.opName();
    }

    @Override
    public NodeDef getNodeWithNameFromGraph(GraphDef graph, String name) {
        for(int i = 0; i < graph.getNodeCount(); i++) {
            val node = graph.getNode(i);
            if(node.getName().equals(name))
                return node;
        }

        return null;
    }

    @Override
    public void mapProperty(String name, DifferentialFunction on, NodeDef node, GraphDef graph, SameDiff sameDiff, Map<String, Map<String, PropertyMapping>> propertyMappingsForFunction) {
        if(node == null) {
            throw new ND4JIllegalStateException("No node found for name " + name);
        }


        val mapping = propertyMappingsForFunction.get(getOpType(node)).get(name);
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(on);


        if(mapping.getTfInputPosition() != null && mapping.getTfInputPosition() < node.getInputCount()) {
            int tfMappingIdx = mapping.getTfInputPosition();
            if(tfMappingIdx < 0)
                tfMappingIdx += node.getInputCount();

            val input = node.getInput(tfMappingIdx);
            val inputNode = TFGraphMapper.getInstance().getNodeWithNameFromGraph(graph,input);
            INDArray arr = getArrayFrom(inputNode,graph);
            if(arr == null) {
                arr = sameDiff.getArrForVarName(input);
            }

            if(arr == null && inputNode != null) {
                sameDiff.addPropertyToResolve(on,name);
                sameDiff.addVariableMappingForField(on,name,inputNode.getName());
                return;
            }
            else if(inputNode == null) {
                sameDiff.addAsPlaceHolder(input);
                return;
            }

            val field = fields.get(name);
            val type = field.getType();
            if(type.equals(int[].class)) {
                on.setValueFor(field,arr.data().asInt());
            }
            else if(type.equals(int.class) || type.equals(long.class) || type.equals(Long.class) || type.equals(Integer.class)) {
                if(mapping.getShapePosition() != null) {
                    on.setValueFor(field,arr.size(mapping.getShapePosition()));
                }
                else
                    on.setValueFor(field,arr.getInt(0));

            }
            else if(type.equals(float.class) || type.equals(double.class) || type.equals(Float.class) || type.equals(Double.class)) {
                on.setValueFor(field,arr.getDouble(0));
            }


        }
        else {
            val tfMappingAttrName = mapping.getTfAttrName();
            if(tfMappingAttrName == null) {
                return;
            }

            if(!node.containsAttr(tfMappingAttrName)) {
                return;
            }


            val attr = node.getAttrOrThrow(tfMappingAttrName);
            val type = attr.getType();
            if(fields == null) {
                throw new ND4JIllegalStateException("No fields found for op [" + mapping + "]");
            }

            if(mapping.getPropertyNames() == null) {
                throw new ND4JIllegalStateException("no property found for [" + name + "] in op [" + on.opName()+"]");
            }

            val field = fields.get(mapping.getPropertyNames()[0]);

            Object valueToSet = null;
            switch(type) {
                case DT_BOOL:
                    valueToSet = attr.getB();
                    break;
                case DT_INT8:
                    valueToSet = attr.getI();
                    break;
                case DT_INT16:
                    valueToSet = attr.getI();
                    break;
                case DT_INT32:
                    valueToSet = attr.getI();
                    break;
                case DT_FLOAT:
                    valueToSet = attr.getF();
                    break;
                case DT_DOUBLE:
                    valueToSet = attr.getF();
                    break;
                case DT_STRING:
                    valueToSet = attr.getS();
                    break;
                case DT_INT64:
                    valueToSet = attr.getI();
                    break;


            }

            if(field != null && valueToSet != null)
                on.setValueFor(field,valueToSet);
        }
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isPlaceHolderNode(NodeDef node) {
        return node.getOp().startsWith("Placeholder");
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public void dumpBinaryProtoAsText(File inputFile, File outputFile) {
        try {
            GraphDef graphDef = GraphDef.parseFrom(new BufferedInputStream(new FileInputStream(inputFile)));
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFile,true));
            for(NodeDef node : graphDef.getNodeList()) {
                bufferedWriter.write(node.toString());
            }

            bufferedWriter.flush();
            bufferedWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public long[] getShapeFromAttr(AttrValue attr) {
        return shapeFromShapeProto(attr.getShape());
    }

    @Override
    public Map<String, AttrValue> getAttrMap(NodeDef nodeDef) {
        return nodeDef.getAttrMap();
    }

    @Override
    public String getName(NodeDef nodeDef) {
        return nodeDef.getName();
    }

    @Override
    public boolean alreadySeen(NodeDef nodeDef) {
        return seenNodes.contains(nodeDef.getName());
    }

    @Override
    public boolean isVariableNode(NodeDef nodeDef) {
        boolean isVar = nodeDef.getOp().startsWith("VariableV") || nodeDef.getOp().equalsIgnoreCase("const");
        return isVar;
    }

    @Override
    public boolean shouldSkip(NodeDef opType) {
        if(opType == null)
            return true;

        boolean endsWithRead = opType.getName().endsWith("/read");
        boolean isReductionIndices = opType.getOp().endsWith("/reduction_indices");
        return  endsWithRead  || isReductionIndices;
    }

    @Override
    public boolean hasShape(NodeDef nodeDef) {
        return nodeDef.containsAttr(SHAPE_KEY);
    }

    @Override
    public long[] getShape(NodeDef nodeDef) {
        return getShapeFromAttr(nodeDef.getAttrOrThrow(SHAPE_KEY));
    }

    @Override
    public INDArray getArrayFrom(NodeDef nodeDef, GraphDef graph) {
        if(nodeDef == null) {
            return null;
        }

        return getNDArrayFromTensor(nodeDef.getName(),nodeDef, graph);
    }

    @Override
    public String getOpType(NodeDef nodeDef) {
        return nodeDef.getOp();
    }

    /**
     *
     * @param graphDef
     * @return
     */
    @Override
    public List<NodeDef> getNodeList(GraphDef graphDef) {
        return graphDef.getNodeList();
    }

    /**
     *
     * @param name the tensorflow or onnx name
     * @return
     */
    @Override
    public DifferentialFunction getMappedOp(String name) {
        return DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(name);
    }


    /**
     * Map a tensorflow node name
     * to the samediff equivalent
     * for import
     * @param name the name to change
     * @return the input tensorflow name
     */
    public String getNodeName(String name) {
        //tensorflow adds colons to the end of variables representing input index, this strips those off
        String ret = name;
        if(ret.startsWith("^"))
            ret = ret.substring(1);
        if(ret.endsWith("/read")) {
            ret = ret.replace("/read","");
        }
        return ret;
    }

    public boolean isControlDependency(String name){
        return name.startsWith("^");
    }



    @Override
    public Map<String, NodeDef> variablesForGraph(GraphDef graphDef) {
        Map<String,NodeDef> ret = new LinkedHashMap<>();
        List<NodeDef> nodeList = graphDef.getNodeList();
        for(NodeDef nodeDef : nodeList) {
            if(nodeDef.getName().endsWith("/read")) {
                continue;
            }


            val name = translateToSameDiffName(nodeDef.getName(), nodeDef);
            ret.put(name,nodeDef);
        }

        return ret;
    }

    @Override
    public String translateToSameDiffName(String name, NodeDef node) {
        if(isVariableNode(node) || isPlaceHolder(node)) {
            return name;
        }

        StringBuilder stringBuilder = new StringBuilder();
        //strip arg number
        if(name.contains(":")) {
            name = name.substring(0,name.lastIndexOf(':'));
            stringBuilder.append(name);
        }
        else {
            stringBuilder.append(name);
        }


        return stringBuilder.toString();
    }


    @Override
    public Message.Builder getNewGraphBuilder() {
        return GraphDef.newBuilder();
    }

    @Override
    public GraphDef parseGraphFrom(byte[] inputStream) throws IOException {
        return GraphDef.parseFrom(inputStream);
    }

    @Override
    public GraphDef parseGraphFrom(InputStream inputStream) throws IOException {
        return GraphDef.parseFrom(inputStream);
    }

    protected void importCondition(String conditionName, NodeDef tfNode, ImportState<GraphDef,NodeDef> importState) {
        /**
         * Cond structure:
         *
         */
    }

    @Override
    public void mapNodeType(NodeDef tfNode, ImportState<GraphDef,NodeDef> importState) {
        if (shouldSkip(tfNode) || alreadySeen(tfNode) || isVariableNode(tfNode)) {
            return;
        }

        val nodeName = tfNode.getName();

        val diff = importState.getSameDiff();
        if (isVariableNode(tfNode)) {
            List<Long> dimensions = new ArrayList<>();
            Map<String, AttrValue> attributes = getAttrMap(tfNode);
            if (attributes.containsKey(VALUE_ATTR_KEY)) {
                diff.var(getName(tfNode),getArrayFrom(tfNode,importState.getGraph()));
            }
            else if (attributes.containsKey(SHAPE_KEY)) {
                AttrValue shape = attributes.get(SHAPE_KEY);
                long[] shapeArr = getShapeFromAttr(shape);
                int dims = shapeArr.length;
                if (dims > 0) {
                    // even vector is 2d in nd4j
                    if (dims == 1)
                        dimensions.add(1L);

                    for (int e = 0; e < dims; e++) {
                        // TODO: eventually we want long shapes :(
                        dimensions.add(getShapeFromAttr(shape)[e]);
                    }
                }
            }
        }

        else if(isPlaceHolder(tfNode)) {
            val vertexId = diff.getVariable(getName(tfNode));
            diff.addAsPlaceHolder(vertexId.getVarName());
        }
        else {
            val opName = tfNode.getOp();

            // FIXME: early draft
            // conditional import
            /*
            if (nodeName.startsWith("cond") && nodeName.contains("/")) {
                val str = nodeName.replaceAll("/.*$","");
                importCondition(str, tfNode, importState);

                seenNodes.add(nodeName);
                return;
            } else if (nodeName.startsWith("while")) {
                // while loop import

                return;
            }
            */

            val differentialFunction = DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(opName);
            if(differentialFunction == null) {
                throw new ND4JIllegalStateException("No tensorflow op found for " + opName + " possibly missing operation class?");
            }
            try {
                val newInstance = differentialFunction.getClass().newInstance();
                val args = new SDVariable[tfNode.getInputCount()];
                newInstance.setOwnName(tfNode.getName());

                for(int i = 0; i < tfNode.getInputCount(); i++) {
                    String inName = tfNode.getInput(i);
                    boolean controlDep = isControlDependency(inName);
                    String name = getNodeName(inName);
                    args[i] = diff.getVariable(name);
                    if(args[i] == null) {
                        args[i] = diff.var(name, (LongShapeDescriptor) null,new ZeroInitScheme('f'));
                        diff.addAsPlaceHolder(args[i].getVarName());
                    }

                    /**
                     * Note here that we are associating
                     * the output/result variable
                     * with its inputs and notifying
                     * the variable that it has a place holder argument
                     * it should resolve before trying to execute
                     * anything.
                     */
                    if(diff.isPlaceHolder( args[i].getVarName())) {
                        diff.putPlaceHolderForVariable(args[i].getVarName(), name);
                    }
                }



                diff.addArgsFor(args,newInstance);
                newInstance.setSameDiff(importState.getSameDiff());

                newInstance.initFromTensorFlow(tfNode,diff,getAttrMap(tfNode),importState.getGraph());
                mapProperties(newInstance,tfNode,importState.getGraph(),importState.getSameDiff(),newInstance.mappingsForFunction());
                importState.getSameDiff().putFunctionForId(newInstance.getOwnName(),newInstance);
                //ensure we can track node name to function instance later.
                diff.setBaseNameForFunctionInstanceId(tfNode.getName(),newInstance);
                diff.addVarNameForImport(tfNode.getName());

            } catch (Exception e) {
                log.error("Failed with [{}]", opName);
                throw new RuntimeException(e);
            }

        }
    }


    /**
     * Calls {@link #initFunctionFromProperties(DifferentialFunction, Map, NodeDef, GraphDef)}
     * using {@link DifferentialFunction#tensorflowName()}
     * @param on the function to use init on
     * @param attributesForNode the attributes for the node
     * @param node
     * @param graph
     */
    public void initFunctionFromProperties(DifferentialFunction on, Map<String, AttrValue> attributesForNode, NodeDef node, GraphDef graph) {
        initFunctionFromProperties(on.tensorflowName(),on,attributesForNode,node,graph);
    }

    /**
     * Init a function's attributes
     * @param mappedTfName the tensorflow name to pick (sometimes ops have multiple names
     * @param on the function to map
     * @param attributesForNode the attributes for the node
     * @param node
     * @param graph
     */
    public void initFunctionFromProperties(String mappedTfName, DifferentialFunction on, Map<String, AttrValue> attributesForNode, NodeDef node, GraphDef graph) {
        val properties = on.mappingsForFunction();
        val tfProperties = properties.get(mappedTfName);
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(on);
        val attributeAdapters = on.attributeAdaptersForFunction();

        // if there's no properties announced for this function - just return
        if (tfProperties == null)
            return;

        //Can't execute in just any order: sometimes there are dependencies between attribute mappings
        //For example, conv2d strides depend on data format -> need to map data format before mapping strides
        //Solution: map nodes without adapters before nodes with adapters. This doesn't guarantee we'll always be
        // mapping in the right order (for example, we might have adapter(x) depends on adapter(y)) but it should catch most cases
        Map<String,PropertyMapping> map;
        if(attributeAdapters == null || !attributeAdapters.containsKey(mappedTfName)) {
            map = tfProperties;
        } else {
            map = new LinkedHashMap<>();
            for (Map.Entry<String, PropertyMapping> e : tfProperties.entrySet()) {
                if (!attributeAdapters.get(mappedTfName).containsKey(e.getKey())) {
                    //No adapter for this attribute
                    map.put(e.getKey(), e.getValue());
                }
            }
            for (Map.Entry<String, PropertyMapping> e : tfProperties.entrySet()) {
                if (!map.containsKey(e.getKey())) {
                    //Not added on first pass -> must have attribute mapper
                    map.put(e.getKey(), e.getValue());
                }
            }
        }

        for(Map.Entry<String,PropertyMapping> entry : map.entrySet()){
            val tfAttrName = entry.getValue().getTfAttrName();
            val currentField = fields.get(entry.getKey());

            AttributeAdapter adapter = null;
            if(attributeAdapters != null && !attributeAdapters.isEmpty()) {
                val mappers = attributeAdapters.get(mappedTfName);
                val adapterFor = mappers.get(entry.getKey());
                adapter = adapterFor;
            }


            if(tfAttrName != null) {
                if(currentField == null) {
                    continue;
                }

                if(attributesForNode.containsKey(tfAttrName)) {
                    val attr = attributesForNode.get(tfAttrName);
                    switch (attr.getValueCase()) {
                        case B:
                            if (adapter != null) {
                                adapter.mapAttributeFor(attr.getB(), currentField, on);
                            }
                            break;
                        case F: break;
                        case FUNC: break;
                        case S:
                            val setString = attr.getS().toStringUtf8();
                            if(adapter != null) {
                                adapter.mapAttributeFor(setString,currentField,on);
                            }
                            else
                                on.setValueFor(currentField,setString);
                            break;
                        case I:
                            val setInt = (int) attr.getI();
                            if(adapter != null) {
                                adapter.mapAttributeFor(setInt,currentField,on);
                            }
                            else
                                on.setValueFor(currentField,setInt);
                            break;
                        case SHAPE:
                            val shape = attr.getShape().getDimList();
                            int[] dimsToSet = new int[shape.size()];
                            for(int i = 0; i < dimsToSet.length; i++) {
                                dimsToSet[i] = (int) shape.get(i).getSize();
                            }

                            if(adapter != null) {
                                adapter.mapAttributeFor(dimsToSet,currentField,on);
                            }

                            else
                                on.setValueFor(currentField,dimsToSet);
                            break;
                        case VALUE_NOT_SET:break;
                        case PLACEHOLDER: break;
                        case LIST:
                            val setList = attr.getList();
                            if(!setList.getIList().isEmpty()) {
                                val intList = Ints.toArray(setList.getIList());
                                if(adapter != null) {
                                    adapter.mapAttributeFor(intList,currentField,on);
                                }
                                else
                                    on.setValueFor(currentField,intList);
                            }
                            else if(!setList.getBList().isEmpty()) {
                                break;
                            }
                            else if(!setList.getFList().isEmpty()) {
                                val floats = Floats.toArray(setList.getFList());
                                if(adapter != null) {
                                    adapter.mapAttributeFor(floats,currentField,on);
                                }

                                else
                                    on.setValueFor(currentField,floats);
                                break;
                            }
                            else if(!setList.getFuncList().isEmpty()) {
                                break;
                            }
                            else if(!setList.getTensorList().isEmpty()) {
                                break;
                            }
                            break;
                        case TENSOR:
                            val tensorToGet = TFGraphMapper.getInstance().mapTensorProto(attr.getTensor());
                            if(adapter != null) {
                                adapter.mapAttributeFor(tensorToGet,currentField,on);
                            }
                            else
                                on.setValueFor(currentField,tensorToGet);
                            break;
                        case TYPE:
                            if (adapter != null) {
                                adapter.mapAttributeFor(attr.getType(), currentField, on);
                            }
                            break;
                    }
                }
            }

            else if(entry.getValue().getTfInputPosition() != null) {


                int position = entry.getValue().getTfInputPosition();
                if(position < 0) {
                    position += node.getInputCount();
                }

                val inputFromNode = TFGraphMapper.getInstance().getNodeWithNameFromGraph(graph,node.getInput(position));
                INDArray tensor = inputFromNode != null ? TFGraphMapper.getInstance().getNDArrayFromTensor("value",inputFromNode,graph) : null;
                if(tensor == null) {
                    tensor = on.getSameDiff().getArrForVarName(getNodeName(node.getInput(position)));
                }


                if(tensor != null) {
                    //use adapter instead of direct mapping just like above
                    if(adapter != null) {
                        adapter.mapAttributeFor(tensor,currentField,on);
                    }
                    else {
                        if(currentField.getType().equals(int[].class)) {
                            on.setValueFor(currentField,tensor.data().asInt());
                        }
                        else if(currentField.getType().equals(double[].class)) {
                            on.setValueFor(currentField,tensor.data().asDouble());

                        }
                        else if(currentField.getType().equals(float[].class)) {
                            on.setValueFor(currentField,tensor.data().asFloat());

                        }
                        else if(currentField.getType().equals(INDArray.class)) {
                            on.setValueFor(currentField,tensor);
                        }
                        else if(currentField.getType().equals(int.class)) {
                            on.setValueFor(currentField,tensor.getInt(0));
                        }
                        else if(currentField.getType().equals(double.class)) {
                            on.setValueFor(currentField,tensor.getDouble(0));
                        }
                        else if(currentField.getType().equals(float.class)) {
                            on.setValueFor(currentField,tensor.getFloat(0));
                        }
                    }
                } else {
                    on.getSameDiff().addPropertyToResolve(on,entry.getKey());
                }
            }
        }
    }


    @Override
    public org.nd4j.linalg.api.buffer.DataType dataTypeForTensor(NodeDef tensorProto) {
        if(!tensorProto.containsAttr("dtype") && !tensorProto.containsAttr("Tidx") && !tensorProto.containsAttr("T"))
            return org.nd4j.linalg.api.buffer.DataType.UNKNOWN;

        val type = tensorProto.containsAttr("dtype") ? tensorProto.getAttrOrThrow("dtype").getType()
                : tensorProto.containsAttr("T") ? tensorProto.getAttrOrThrow("T").getType() : tensorProto
                .getAttrOrThrow("Tidx").getType();
        switch(type) {
            case DT_DOUBLE: return org.nd4j.linalg.api.buffer.DataType.DOUBLE;
            case DT_INT32:
            case DT_INT64: return org.nd4j.linalg.api.buffer.DataType.INT;
            case DT_FLOAT: return org.nd4j.linalg.api.buffer.DataType.FLOAT;
            case DT_BFLOAT16: return org.nd4j.linalg.api.buffer.DataType.HALF;
            default: return org.nd4j.linalg.api.buffer.DataType.UNKNOWN;
        }
    }

    @Override
    public boolean unknownTypeNodeImportable(NodeDef tensorProto) {
        DataType dt = null;
        if(tensorProto.containsAttr("dtype")){
            dt = tensorProto.getAttrOrThrow("dtype").getType();
        } else if(tensorProto.containsAttr("T")){
            dt = tensorProto.getAttrOrThrow("T").getType();
        } else if(tensorProto.containsAttr("Tidx")){
            dt = tensorProto.getAttrOrThrow("Tidx").getType();
        }

        return dt == DataType.DT_BOOL;
    }

    @Override
    public boolean isStringType(NodeDef tensorProto){
        DataType dt = null;
        if(tensorProto.containsAttr("dtype")){
            dt = tensorProto.getAttrOrThrow("dtype").getType();
        } else if(tensorProto.containsAttr("T")){
            dt = tensorProto.getAttrOrThrow("T").getType();
        } else if(tensorProto.containsAttr("Tidx")){
            dt = tensorProto.getAttrOrThrow("Tidx").getType();
        }

        return dt == DataType.DT_STRING || dt == DataType.DT_STRING_REF;
    }


    @Override
    public String getAttrValueFromNode(NodeDef nodeDef, String key) {
        return nodeDef.getAttrOrThrow(key).getS().toStringUtf8();
    }

    @Override
    public long[] getShapeFromAttribute(AttrValue attrValue) {
        TensorShapeProto shape = attrValue.getShape();
        long[] ret = new long[shape.getDimCount()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = (int) shape.getDim(i).getSize();
        }
        return ret;
    }

    @Override
    public boolean isPlaceHolder(NodeDef nodeDef) {
        return nodeDef.getOp().startsWith("Placeholder");
    }

    @Override
    public boolean isConstant(NodeDef nodeDef) {
        return nodeDef.getOp().startsWith("Const");
    }

    @Override
    public List<String> getControlDependencies(NodeDef node){
        int numInputs = node.getInputCount();
        if(numInputs == 0)
            return null;

        List<String> out = null;
        for( int i=0; i<numInputs; i++ ){
            String in = node.getInput(i);
            if(isControlDependency(in)){
                if(out == null)
                    out = new ArrayList<>();
                out.add(getNodeName(in));       //Remove "^" prefix
            }
        }
        return out;
    }

    @Override
    public  INDArray getNDArrayFromTensor(String tensorName, NodeDef node, GraphDef graph) {
        //placeholder of some kind
        if(!node.getAttrMap().containsKey("value")) {
            return null;
        }

        val tfTensor = node.getAttrOrThrow("value").getTensor();
        return mapTensorProto(tfTensor);
    }



    public INDArray mapTensorProto(TensorProto tfTensor) {
        // building shape first
        int dims = tfTensor.getTensorShape().getDimCount();
        long[] arrayShape = null;
        List<Integer> dimensions = new ArrayList<>();
        for (int e = 0; e < dims; e++) {
            // TODO: eventually we want long shapes :(
            int dim = (int) tfTensor.getTensorShape().getDim(e).getSize();
            dimensions.add(dim);
        }



        arrayShape = ArrayUtil.toLongArray(Ints.toArray(dimensions));

        if (tfTensor.getDtype() == DataType.DT_INT32 || tfTensor.getDtype() == DataType.DT_INT16 || tfTensor.getDtype() == DataType.DT_INT8) {
            // valueOf
            if (tfTensor.getIntValCount() == 1 || ArrayUtil.prod(arrayShape) == 1) {
                //straight zero case
                if(tfTensor.getIntValCount() < 1)
                    return Nd4j.scalar( ArrayOptionsHelper.convertToDataType(tfTensor.getDtype()), 0);

                //should be scalar otherwise
                int val = tfTensor.getIntVal(0);

                if (arrayShape == null || arrayShape.length == 0)
                    return Nd4j.scalar( ArrayOptionsHelper.convertToDataType(tfTensor.getDtype()), val);

                return Nd4j.valueArrayOf(arrayShape, val, ArrayOptionsHelper.convertToDataType(tfTensor.getDtype()));
            } else if (tfTensor.getInt64ValCount() > 0) {
                val jArray = new int[tfTensor.getIntValCount()];
                for (int e = 0; e < tfTensor.getIntValCount(); e++) {
                    jArray[e] = tfTensor.getIntVal(e);
                }

                // TF arrays are always C
                return Nd4j.create(Nd4j.createTypedBuffer(jArray, ArrayOptionsHelper.convertToDataType(tfTensor.getDtype())), arrayShape, Nd4j.getStrides(arrayShape, 'c'), 0, 'c', ArrayOptionsHelper.convertToDataType(tfTensor.getDtype()));
            } else {
                // FIXME: INT bytebuffers should be converted to floating point
                //throw new UnsupportedOperationException("To be implemented yet");
                long length = ArrayUtil.prodLong(arrayShape);
                // binary representation
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val fb = bb.order(ByteOrder.nativeOrder()).asIntBuffer();
                val fa = new int[fb.capacity()];
                for (int e = 0; e < fb.capacity(); e++)
                    fa[e] = fb.get(e);

                if (fa.length == 0)
                    return Nd4j.empty(ArrayOptionsHelper.convertToDataType(tfTensor.getDtype()));
                    //throw new ND4JIllegalStateException("Can't find Tensor values! Probably you've forgot to freeze graph before saving?");

                if (fa.length == 1)
                    return Nd4j.scalar(ArrayOptionsHelper.convertToDataType(tfTensor.getDtype()), fa[0]);

                if (arrayShape.length == 1)
                    return Nd4j.create(fa, new long[]{fa.length}, new long[]{1}, 'c', ArrayOptionsHelper.convertToDataType(tfTensor.getDtype()));

                val array = Nd4j.create(Nd4j.createTypedBuffer(fa, ArrayOptionsHelper.convertToDataType(tfTensor.getDtype())), arrayShape, Nd4j.getStrides(arrayShape, 'c'), 0, 'c', ArrayOptionsHelper.convertToDataType(tfTensor.getDtype()));
                //log.debug("SUM1: {}", array.sumNumber());
                //log.debug("Data: {}", Arrays.toString(array.data().asFloat()));
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_FLOAT) {
            if (tfTensor.getFloatValCount() == 1 || ArrayUtil.prod(arrayShape) == 1) {
                //straight zero case
                if(tfTensor.getFloatValCount() < 1)
                    return Nd4j.scalar(org.nd4j.linalg.api.buffer.DataType.FLOAT, 0.0f);


                float val = tfTensor.getFloatVal(0);

                if (arrayShape == null || arrayShape.length == 0)
                    arrayShape = new long[]{};

                INDArray array = Nd4j.valueArrayOf(arrayShape, val, org.nd4j.linalg.api.buffer.DataType.FLOAT);
                return array;
            } else if (tfTensor.getFloatValCount() > 0) {
                float[] jArray = new float[tfTensor.getFloatValCount()];
                for (int e = 0; e < tfTensor.getFloatValCount(); e++) {
                    jArray[e] = tfTensor.getFloatVal(e);
                }

                INDArray array = Nd4j.create(Nd4j.createTypedBuffer(jArray, org.nd4j.linalg.api.buffer.DataType.FLOAT), arrayShape, Nd4j.getStrides(arrayShape), 0, 'c');
                return array;
            } else if (tfTensor.getTensorContent().size() > 0){
                // binary representation
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val fb = bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
                val fa = new float[fb.capacity()];
                for (int e = 0; e < fb.capacity(); e++)
                    fa[e] = fb.get(e);

                if (fa.length == 0)
                    throw new ND4JIllegalStateException("Can't find Tensor values! Probably you've forgot to freeze graph before saving?");

                if (fa.length == 1)
                    return Nd4j.scalar(org.nd4j.linalg.api.buffer.DataType.FLOAT, fa[0]);

                if (arrayShape.length == 1)
                    return Nd4j.create(fa, new long[]{fa.length}, new long[]{1}, 'c', org.nd4j.linalg.api.buffer.DataType.FLOAT);

                val array = Nd4j.create(fa, arrayShape, Nd4j.getStrides(arrayShape, 'c'), 'c', org.nd4j.linalg.api.buffer.DataType.FLOAT);
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_DOUBLE) {
            if (tfTensor.getDoubleValCount() == 1 || ArrayUtil.prod(arrayShape) == 1) {
                //straight zero case
                if(tfTensor.getDoubleValCount() < 1)
                    return Nd4j.trueScalar(0.0);

                double val = tfTensor.getDoubleVal(0);
                INDArray array = Nd4j.trueScalar(val);
                return array;
            } else if (tfTensor.getDoubleValCount() > 0) {
                val jArray = new double[tfTensor.getDoubleValCount()];
                for (int e = 0; e < tfTensor.getDoubleValCount(); e++) {
                    jArray[e] =  tfTensor.getDoubleVal(e);
                }

                // TF arrays are always C
                val array = Nd4j.create(jArray, arrayShape, Nd4j.getStrides(arrayShape, 'c'), 'c', org.nd4j.linalg.api.buffer.DataType.DOUBLE);
                return array;
            } else if (tfTensor.getTensorContent().size() > 0) {
                // binary representation
                //DataBuffer buffer = Nd4j.createBuffer(tfTensor.getTensorContent().asReadOnlyByteBuffer(), DataType.FLOAT, (int) length);
                //INDArray array = Nd4j.createArrayFromShapeBuffer(buffer, Nd4j.getShapeInfoProvider().createShapeInformation(arrayShape, 'c'));

                // binary representation
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val fb = bb.order(ByteOrder.nativeOrder()).asDoubleBuffer();
                val da = new double[fb.capacity()];
                for (int e = 0; e < fb.capacity(); e++)
                    da[e] = fb.get(e);

                if (da.length == 0)
                    throw new ND4JIllegalStateException("Can't find Tensor values! Probably you've forgot to freeze graph before saving?");

                if (da.length == 1)
                    return Nd4j.trueScalar(da[0]);

                if (arrayShape.length == 1)
                    return Nd4j.trueVector(da);

                val array = Nd4j.create(da, arrayShape, 0, 'c');
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_INT64) {
            if (tfTensor.getInt64ValCount() == 1 || ArrayUtil.prod(arrayShape) == 1) {
                //straight zero case
                if (tfTensor.getInt64ValCount() < 1)
                    return Nd4j.trueScalar(0.0);

                double val = (double) tfTensor.getInt64Val(0);
                INDArray array = Nd4j.trueScalar(val);
                return array;
            } else if (tfTensor.getInt64ValCount() > 0) {
                val jArray = new long[tfTensor.getInt64ValCount()];
                for (int e = 0; e < tfTensor.getInt64ValCount(); e++) {
                    jArray[e] = tfTensor.getInt64Val(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(Nd4j.createTypedBuffer(jArray, org.nd4j.linalg.api.buffer.DataType.LONG), arrayShape, Nd4j.getStrides(arrayShape, 'c'),0, 'c', org.nd4j.linalg.api.buffer.DataType.LONG);
                return array;
            } else if (tfTensor.getTensorContent().size() > 0) {
                //throw new UnsupportedOperationException("To be implemented yet");
                //Mapping INT bytebuffers should be converted to floating point
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val lb = bb.order(ByteOrder.nativeOrder()).asLongBuffer();
                val fa = new long[lb.capacity()];
                for (int e = 0; e < lb.capacity(); e++)
                    fa[e] = lb.get(e);
                if (fa.length == 0)
                    throw new ND4JIllegalStateException("Can't find Tensor values! Probably you've forgot to freeze graph before saving?");

                if (fa.length == 1)
                    return Nd4j.trueScalar(fa[0]);

                if (arrayShape.length == 1)
                    return Nd4j.trueVector(fa);

                val array = Nd4j.create(Nd4j.createTypedBuffer(fa, org.nd4j.linalg.api.buffer.DataType.LONG), arrayShape, Nd4j.getStrides(arrayShape, 'c'),  0, 'c', org.nd4j.linalg.api.buffer.DataType.LONG);
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_BOOL){
            if (tfTensor.getBoolValCount() == 1 || ArrayUtil.prod(arrayShape) == 1){
                //straight zero case
                if(tfTensor.getBoolValCount() < 1)
                    return Nd4j.scalar(false);

                val val = tfTensor.getBoolVal(0);
                val arr = Nd4j.scalar(val);
                return arr;
            } else if (tfTensor.getBoolValCount() > 0) {
                val jArray = new boolean[tfTensor.getBoolValCount()];
                for (int e = 0; e < tfTensor.getBoolValCount(); e++) {
                    jArray[e] = tfTensor.getBoolVal(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(Nd4j.createTypedBuffer(jArray, org.nd4j.linalg.api.buffer.DataType.BOOL), arrayShape, Nd4j.getStrides(arrayShape, 'c'), 0,  'c', org.nd4j.linalg.api.buffer.DataType.BOOL);
                return array;
            } else if (tfTensor.getTensorContent().size() > 0) {
                throw new UnsupportedOperationException("Not yet implemented for DataType.DT_BOOL");
            }
        }  else {
            throw new UnsupportedOperationException("Unknown dataType found: [" + tfTensor.getDtype() + "]");
        }

        throw new ND4JIllegalStateException("Invalid method state");
    }

    @Override
    public long[] getShapeFromTensor(NodeDef tensorProto) {
        if(tensorProto.containsAttr("shape")) {
            return shapeFromShapeProto(tensorProto.getAttrOrThrow("shape").getShape());

        }
        //yet to be determined shape, or tied to an op where output shape is dynamic
        else if(!tensorProto.containsAttr("value")) {
            return null;

        }
        else
            return shapeFromShapeProto(tensorProto.getAttrOrThrow("value").getTensor().getTensorShape());
    }

    @Override
    public Set<String> opsToIgnore() {
        return graphMapper;
    }


    @Override
    public String getInputFromNode(NodeDef node, int index) {
        return node.getInput(index);
    }

    @Override
    public int numInputsFor(NodeDef nodeDef) {
        return nodeDef.getInputCount();
    }

    private long[] shapeFromShapeProto(TensorShapeProto tensorShapeProto) {
        long[] shape = new long[tensorShapeProto.getDimList().size()];
        for(int i = 0; i < shape.length; i++) {
            shape[i] =  tensorShapeProto.getDim(i).getSize();
        }

        return shape;
    }


    /**
     * Returns the node for an if statement
     * @param from the starting node (a merge node that represents a conditional)
     * @param graph the graph to search
     * @return an import state representing the nodes for each scope
     */
    public IfImportState nodesForIf(NodeDef from, GraphDef graph) {
        //Assume we start with a switch statement
        int currNodeIndex = graph.getNodeList().indexOf(from);
        val trueDefName = from.getInput(1);
        val falseDefName = from.getInput(0);
        val scopeId = UUID.randomUUID().toString();
        val scopeName = scopeId + "-" + trueDefName.substring(0,trueDefName.indexOf("/"));
        val trueDefScopeName = scopeName + "-true-scope";
        val falseDefScopeName = scopeName + "-false-scope";


        boolean onFalseDefinition = true;
        //start with the true
        boolean onTrueDefinition = false;

        List<NodeDef> falseBodyNodes = new ArrayList<>();
        List<NodeDef> trueBodyNodes = new ArrayList<>();
        List<NodeDef> conditionNodes = new ArrayList<>();
        Set<String> seenNames = new LinkedHashSet<>();
        /**
         * Accumulate a list backwards to get proper ordering.
         *
         */
        for(int i = currNodeIndex; i >= 0; i--) {
            //switch to false names
            if(graph.getNode(i).getName().equals(trueDefName)) {
                onFalseDefinition = false;
                onTrueDefinition = true;
            }

            //on predicate now
            if(graph.getNode(i).getName().contains("pred_id")) {
                onTrueDefinition = false;
            }
            //don't readd the same node, this causes a stackoverflow
            if(onTrueDefinition  && !graph.getNode(i).equals(from)) {
                trueBodyNodes.add(graph.getNode(i));
            }
            else if(onFalseDefinition && !graph.getNode(i).equals(from)) {
                falseBodyNodes.add(graph.getNode(i));
            }
            //condition scope now
            else {
                val currNode = graph.getNode(i);
                if(currNode.equals(from))
                    continue;

                //break only after bootstrapping the first node (the predicate id node)
                if(!seenNames.contains(graph.getNode(i).getName()) && !graph.getNode(i).getName().contains("pred_id")) {
                    break;
                }

                /**
                 * Continuously add inputs seen for each node in the sub graph that occurs.
                 * Starting from the predicate id, any node that has inputs in the condition scope
                 * are by definition within the scope. Any node not encountered after that is considered out of scope.
                 * This means we break.
                 */
                for(int inputIdx = 0; inputIdx < currNode.getInputCount(); inputIdx++) {
                    seenNames.add(currNode.getInput(inputIdx));
                }



                //ensure the "current node" is added as well
                seenNames.add(graph.getNode(i).getName());
                conditionNodes.add(graph.getNode(i));
            }
        }

        /**
         * Since we are going over the graph backwards,
         * we need to reverse the nodes to ensure proper ordering.
         */
        Collections.reverse(falseBodyNodes);
        Collections.reverse(trueBodyNodes);
        Collections.reverse(conditionNodes);


        return IfImportState.builder()
                .condNodes(conditionNodes)
                .falseNodes(falseBodyNodes)
                .trueNodes(trueBodyNodes)
                .conditionBodyScopeName(falseDefScopeName)
                .falseBodyScopeName(falseDefScopeName)
                .trueBodyScopeName(trueDefScopeName)
                .conditionBodyScopeName(scopeName)
                .build();
    }



}
