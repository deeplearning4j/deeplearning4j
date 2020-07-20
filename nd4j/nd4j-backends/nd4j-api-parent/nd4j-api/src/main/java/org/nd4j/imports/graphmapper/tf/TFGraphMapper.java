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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.IOUtils;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMapper;
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMappers;
import org.nd4j.imports.tensorflow.TFImportOverride;
import org.nd4j.imports.tensorflow.TFOpImportFilter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge;
import org.nd4j.shade.guava.primitives.Floats;
import org.nd4j.shade.guava.primitives.Ints;
import org.nd4j.shade.protobuf.Message;
import org.nd4j.shade.protobuf.TextFormat;
import org.tensorflow.framework.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Import a TensorFlow frozen graph in ProtoBuf (.pb) format, to SameDiff
 *
 * @author Alex Black
 */
@Slf4j
public class TFGraphMapper {

    /**
     * @deprecated Use static methods - {@link #importGraph(File)} etc
     */
    @Deprecated
    public static TFGraphMapper getInstance(){
        return new TFGraphMapper();
    }

    /**
     * Import a frozen TensorFlow protobuf (.pb) file from the specified file
     *
     * @param f Frozen TensorFlow model pb file to import
     * @return Imported graph
     */
    public static SameDiff importGraph(@NonNull File f) {
        return importGraph(f, null, null);
    }

    /**
     * Import a frozen TensorFlow protobuf (.pb) file from the specified file, with optional overrides
     *
     * @param f              Frozen TensorFlow model pb file to import
     * @param importOverride Optional import override for specific ops, keyed by op name
     * @param opFilter       Optional filter - ops to exclude/ignore
     * @return Imported graph
     */
    public static SameDiff importGraph(@NonNull File f, Map<String, TFImportOverride> importOverride, TFOpImportFilter opFilter) {
        Preconditions.checkState(f.exists(), "File does not exist: %s", f);
        try (InputStream is = new BufferedInputStream(new FileInputStream(f))) {
            return importGraph(is, importOverride, opFilter);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Import a frozen TensorFlow protobuf (.pb) file, via an input stream
     *
     * @param is Stream for a frozen TensorFlow model pb file to import
     * @return Imported graph
     */
    public static SameDiff importGraph(@NonNull InputStream is) {
        return importGraph(is, null, null);
    }

    /**
     * Import a frozen TensorFlow protobuf file in text format (.pb.txt) file via an input stream, with optional overrides
     *
     * @param is             Stream for a frozen TensorFlow model pb file to import
     * @param importOverride Optional import override for specific ops, keyed by op name
     * @param opFilter       Optional filter - ops to exclude/ignore
     * @return Imported graph
     */
    public static SameDiff importGraphTxt(@NonNull InputStream is, Map<String, TFImportOverride> importOverride, TFOpImportFilter opFilter) {
        GraphDef tfGraph;
        try {
            Message.Builder builder = GraphDef.newBuilder();
            String content = IOUtils.toString(is, StandardCharsets.UTF_8);
            TextFormat.getParser().merge(content, builder);
            tfGraph = (GraphDef) builder.build();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return importGraph(tfGraph, importOverride, opFilter);
    }

    /**
     * Import a frozen TensorFlow protobuf (.pb) file via an input stream, with optional overrides
     *
     * @param is             Stream for a frozen TensorFlow model pb file to import
     * @param importOverride Optional import override for specific ops, keyed by op name
     * @param opFilter       Optional filter - ops to exclude/ignore
     * @return Imported graph
     */
    public static SameDiff importGraph(@NonNull InputStream is, Map<String, TFImportOverride> importOverride, TFOpImportFilter opFilter) {
        GraphDef tfGraph;
        try {
            tfGraph = GraphDef.parseFrom(is);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return importGraph(tfGraph, importOverride, opFilter);
    }

    /**
     * Import a TensorFlow model from a GraphDef
     *
     * @param tfGraph TensorFlow model GraphDef
     * @return Imported model
     */
    public static SameDiff importGraph(@NonNull GraphDef tfGraph) {
        return importGraph(tfGraph, null, null);
    }

    /**
     * Import a TensorFlow model from a GraphDef, with optional import overrides
     *
     * @param tfGraph        TensorFlow model GraphDef
     * @param importOverride Optional import override for specific ops, keyed by op name
     * @param opFilter       Optional filter - ops to exclude/ignore
     * @return Imported model
     */
    public static SameDiff importGraph(@NonNull GraphDef tfGraph, Map<String, TFImportOverride> importOverride, TFOpImportFilter opFilter) {

        /*
        First, build an in-memory representation of the graph that allows us to build the graph incrementally
        If we can build the graph incrementally, we can make sure that the added variables are set up with the correct
        datatype and (once implemented) greedy shape inference
         */
        Set<String> availableToAddSet = new HashSet<>();            //TODO maybe unnecessary?
        Queue<NodeDef> availableToAdd = new LinkedList<>();

        Map<String, NodeDef> remainingNodes = new HashMap<>();          //All other nodes, not in availableToAdd

        Map<String, Set<String>> nodeInputTo = new HashMap<>();     // For op x -> y, x is key, y is value. Note that these are OP names not VARIABLE names

        int nNodes = tfGraph.getNodeCount();

        //First, add any constants, placeholders, and zero-input ops
        SameDiff sd = SameDiff.create();
        for (int i = 0; i < nNodes; i++) {
            NodeDef nd = tfGraph.getNode(i);
            String op = nd.getOp();
            String name = nd.getName();

            int nInputs = nd.getInputCount();

            if ("Const".equals(op) || "Placeholder".equals(op) || nInputs == 0) {
                availableToAdd.add(nd);
                availableToAddSet.add(name);
            } else {
                remainingNodes.put(name, nd);
                for (int in = 0; in < nInputs; in++) {
                    String inOpName = stripControl(nd.getInput(in));
                    inOpName = stripVarSuffix(inOpName);

                    if (!nodeInputTo.containsKey(inOpName)) {
                        nodeInputTo.put(inOpName, new HashSet<String>());
                    }
                    nodeInputTo.get(inOpName).add(name);
                }
            }
        }

        Map<String, String> mergeOpsPostProcess = new HashMap<>();

        //Go through ops in order, and add to the graph
        Map<String, List<String>> constControlDeps = new HashMap<>();        //Key: constant name. Value: control dependencies
        while (!availableToAdd.isEmpty()) {
            NodeDef nd = availableToAdd.remove();
            String name = nd.getName();
            String opName = nd.getOp();
            int nIn = nd.getInputCount();

            availableToAddSet.remove(name);

            log.trace("Adding operation to graph: {} (name={})", opName, name);

            boolean skipCase = false;
            if(opFilter != null && opFilter.skipOp(nd, sd, nd.getAttrMap(), tfGraph)){
                log.debug("Skipping op {} of type {} due to op filter", name, opName);
                //Don't continue at this point - we still need to process what this feeds into...
                skipCase = true;
            } else {
                if (importOverride == null || !importOverride.containsKey(name)) {
                    //Standard case
                    if ("Const".equals(opName)) {
                        //Get array, create a constant
                        TensorProto tfTensor = nd.getAttrOrThrow("value").getTensor();
                        TFTensorMapper m = TFTensorMappers.newMapper(tfTensor);
                        INDArray arr = m.toNDArray();
                        sd.constant(name, arr);
                        int inputCount = nd.getInputCount();
                        if (inputCount > 0) {
                            //Very likely control dependency. i.e., "we must execute op X before the constant is really available to be used"
                            List<String> l = new ArrayList<>(inputCount);
                            for (int i = 0; i < inputCount; i++) {
                                String n = nd.getInput(i);
                                if (!isControlDep(n)) {
                                    throw new IllegalStateException("Found non-control dependency input \"" + n + "\" for constant \"" + name + "\"");
                                }
                                String n2 = stripControl(n);
                                l.add(n2);
                            }
                            constControlDeps.put(name, l);
                        }
                    } else if ("Placeholder".equals(opName) || "PlaceholderWithDefault".equals(opName)) {
                        //TODO support the "WithDefault" array

                        Map<String, AttrValue> attrMap = nd.getAttrMap();
                        boolean shapeAvailable = attrMap.containsKey("shape");
                        long[] shape;
                        if (shapeAvailable) {
                            TensorShapeProto shapeProto = attrMap.get("shape").getShape();
                            shape = shapeFromShapeProto(shapeProto);
                        } else {
                            //Some placeholders don't have any shape restrictions - i.e., accept anything...
                            shape = null;
                        }


                        org.tensorflow.framework.DataType tfDtype = attrMap.get("dtype").getType();
                        org.nd4j.linalg.api.buffer.DataType dt = convertType(tfDtype);
                        sd.placeHolder(name, dt, shape);
                    } else {
                        /*
                        Normal ops. Process in the following order:
                        1. Create the op instance
                        2. Add op to graph
                        3. Import from TF (to set attributes)
                        4. Calculate output dtypes
                        5. Create and add output variables to graph

                        Note: one constraint on this order is that some ops import modify the graph structure.
                        Notable example: concat op - it removes the axis op and converts the value to an iArg
                        https://github.com/eclipse/deeplearning4j/issues/8285
                         */
                        DifferentialFunction dfInstance = DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(opName);
                        Preconditions.checkState(dfInstance != null, "Could not find class for TF Ops: %s", opName);

                        DifferentialFunction df;
                        try {
                            df = dfInstance.getClass().newInstance();
                        } catch (Throwable t) {
                            //Should never happen because function was already created via no-arg constructor earlier
                            throw new RuntimeException(t);
                        }
                        df.setSameDiff(sd);
                        df.setOwnName(name);

                        //Process inputs
                        List<String> inNames = new ArrayList<>(nIn);
                        List<String> controlDeps = null;
                        for (int i = 0; i < nIn; i++) {
                            String origInName = nd.getInput(i);
                            String inName = stripControl(origInName);

                            if(inName.endsWith(":0")){
                                //Strip ":0" suffix. Some ops can depend on placeholders, like "image_tensor:0" but in SameDiff this is a variable called "image_tensor"
                                inName = inName.substring(0, inName.length()-2);
                            }

                            boolean isControlDep = isControlDep(origInName);
                            if (isControlDep) {
                                if (controlDeps == null)
                                    controlDeps = new ArrayList<>();
                                controlDeps.add(inName);
                            }

                            if (!isControlDep) {
                                inNames.add(inName);
                            }

                            //Update Variable.inputsForOp for all variables that feed into this op
                            // Such variables must have already been created, given we process in order
                            Variable v = sd.getVariables().get(inName);

                            if (v == null && df instanceof Merge) {
                                //Edge case for import - we allow merge ops to be added before both inputs are available
                                //This is to break the cycles in loops, otherwise we can't process anything in order
                                mergeOpsPostProcess.put(df.getOwnName(), inName);
                                continue;
                            }

                            if (!isControlDep && (v.getInputsForOp() == null || !v.getInputsForOp().contains(name))) {
                                //May already be present - for example, add(x,x)
                                if (v.getInputsForOp() == null)
                                    v.setInputsForOp(new ArrayList<String>());
                                v.getInputsForOp().add(name);
                            } else if (isControlDep) {
                                if (v.getControlDepsForOp() == null)
                                    v.setControlDepsForOp(new ArrayList<String>());
                                if (!v.getControlDepsForOp().contains(name)) {
                                    v.getControlDepsForOp().add(name);
                                }
                            }
                        }

                        //Create SameDiffOp instance and add to graph
                        SameDiffOp op = SameDiffOp.builder()
                                .name(name)
                                .op(df)
                                .inputsToOp(inNames)
                                //.outputsOfOp(outNames)    //We'll set this later
                                .controlDeps(controlDeps)
                                .build();
                        sd.getOps().put(name, op);


                        Map<String, AttrValue> attrMap = nd.getAttrMap();
                        df.initFromTensorFlow(nd, sd, attrMap, tfGraph);            //TODO REMOVE TFGRAPH ENTIRELY FROM THIS CALL - it encourages hacky and really brittle stuff like input array to attribute conversion

                        //DType calculate for output variables (set/correct if necessary)
                        List<String> newInNames = sd.getOps().get(name).getInputsToOp();        //Just in case import has modified this, like for concat case
                        List<org.nd4j.linalg.api.buffer.DataType> newInDtypes = new ArrayList<>(newInNames.size());
                        if (df instanceof Merge) {
                            //Merge op: as noted elsewhere, we allow merge to be processed when only one of the inputs is available
                            // to break cycles for loops
                            //We know that Merge op has the restriction of the same datatype for both inputs, so we'll
                            SDVariable v1 = sd.getVariable(newInNames.get(0));
                            SDVariable v2 = sd.getVariable(newInNames.get(1));
                            org.nd4j.linalg.api.buffer.DataType dt1 = (v1 == null ? v2.dataType() : v1.dataType());
                            org.nd4j.linalg.api.buffer.DataType dt2 = (v2 == null ? v1.dataType() : v2.dataType());
                            newInDtypes.add(dt1);
                            newInDtypes.add(dt2);
                        } else {
                            for (String s : newInNames) {
                                SDVariable v = sd.getVariable(s);
                                newInDtypes.add(v.dataType());
                            }
                        }

                        List<org.nd4j.linalg.api.buffer.DataType> outDTypes = df.calculateOutputDataTypes(newInDtypes);
                        SDVariable[] outSDVars = new SDVariable[outDTypes.size()];
                        Variable[] outVars = new Variable[outDTypes.size()];
                        List<String> outNames = new ArrayList<>(outDTypes.size());

                        //Create output variables and add to graph
                        for (int i = 0; i < outDTypes.size(); i++) {
                            org.nd4j.linalg.api.buffer.DataType dt = outDTypes.get(i);
                            String varName = name + (i == 0 ? "" : ":" + i);
                            outSDVars[i] = sd.var(varName, VariableType.ARRAY, null, dt, (long[]) null);
                            outNames.add(varName);

                            outVars[i] = Variable.builder()
                                    .name(varName)
                                    .variable(outSDVars[i])
                                    .inputsForOp(null)          //This is updated incrementally as other ops are added
                                    .controlDepsForOp(null)     //Control deps are handled later
                                    .controlDepsForVar(null)
                                    .outputOfOp(name)
                                    .build();

                            sd.getVariables().put(varName, outVars[i]);
                            log.trace("Added variable to graph: {} (output of op {})", varName, name);
                        }
                        sd.getOps().get(name).setOutputsOfOp(outNames);

                        log.trace("Imported op: {} (name={})", opName, name);
                    }
                } else {
                    //Import override case
                    TFImportOverride o = importOverride.get(name);

                    log.debug("Importing op {} using override {}", opName, importOverride);

                    //First, get inputs:
                    List<SDVariable> inputs = new ArrayList<>(nIn);
                    List<SDVariable> controlDeps = null;
                    for (int i = 0; i < nIn; i++) {
                        String inName = nd.getInput(i);
                        boolean controlDep = isControlDep(inName);

                        SDVariable v = sd.getVariable(name);

                        if (controlDep) {
                            if (controlDeps == null)
                                controlDeps = new ArrayList<>();
                            controlDeps.add(v);
                        } else {
                            inputs.add(v);
                        }

                        o.initFromTensorFlow(inputs, controlDeps, nd, sd, nd.getAttrMap(), tfGraph);
                    }
                }
            }


            //Now that we have just added an op (or variable) - check what this feeds into, and see what we can now process
            // as a result
            if (nodeInputTo.containsKey(name)) {
                Set<String> set = nodeInputTo.get(name);
                for (String nextOp : set) {
                    NodeDef nextOpDef = remainingNodes.get(nextOp);
                    if (nextOpDef == null) {
                        if (sd.getOps().containsKey(nextOp)) {
                            //Already processed this.
                            //Almost certainly the close of a loop - like NextIteration -> Merge case
                            continue;
                        }
                        //Should never happen
                        throw new IllegalStateException("Could not find op definition for op to import: " + nextOp);
                    }

                    int nInNext = nextOpDef.getInputCount();
                    boolean allAlreadyInGraph = true;
                    int nonControlSeenCount = 0;
                    for (int i = 0; i < nInNext; i++) {
                        String s = nextOpDef.getInput(i);
                        String inName = stripControl(nextOpDef.getInput(i));

                        if(inName.endsWith(":0")){
                            //Strip ":0" suffix. Some ops can depend on placeholders, like "image_tensor:0" but in SameDiff this is a variable called "image_tensor"
                            inName = inName.substring(0, inName.length()-2);
                        }

//                        log.info("Input: {}, {}", s, inName);

                        if (!sd.hasVariable(inName) && !skipCase) {
//                            log.info("Not found: {} for op {}", inName, nextOpDef.getName());
                            allAlreadyInGraph = false;
                            break;
                        } else if (!isControlDep(s)) {
                            nonControlSeenCount++;
                        }
                    }

                    //Merge ops are an edge case. We'll allow these to be executed with just ONE input, to break
                    // the cycle in loops. In loops, generally we have (Enter, NextIteration) -> Merge, which
                    // of course can't be done if we strictly require all inputs to be available
                    boolean mergeCase = (nonControlSeenCount > 0 && "Merge".equals(nextOpDef.getOp()));

                    if (allAlreadyInGraph || mergeCase) {
                        //Can process this op, add it to the queue for processing
                        if (!availableToAddSet.contains(nextOp)) {
                            //Avoid processing same op multiple times, for repeated inputs to one op, etc
                            availableToAdd.add(nextOpDef);
                            availableToAddSet.add(nextOp);
                            log.trace("Added to processing queue: {} (name={})", nextOpDef.getOp(), nextOp);
                        }
                    }
                }
            }

            //Finally, remove the just processed op from remainingNodes map:
            remainingNodes.remove(name);
        }

        //Post process the control dependencies, if any (done after because dependencies may not exist when imported)
        for (Map.Entry<String, List<String>> e : constControlDeps.entrySet()) {
            String varName = e.getKey();
            List<String> cdOpNames = e.getValue();
            sd.getVariables().get(varName).setControlDeps(cdOpNames);

            for (String s : cdOpNames) {
                SameDiffOp sdo = sd.getOps().get(s);
                if (sdo.getControlDepFor() == null)
                    sdo.setControlDepFor(new ArrayList<String>());
                List<String> l = sdo.getControlDepFor();
                if (!l.contains(s))
                    l.add(varName);
            }
        }

        //Post process the merge ops - all we are missing is a Variable.getInputsForOp().add(mergeOpName);
        for (Map.Entry<String, String> e : mergeOpsPostProcess.entrySet()) {
            Variable v = sd.getVariables().get(e.getValue());
            if (v.getInputsForOp() == null)
                v.setInputsForOp(new ArrayList<String>());
            v.getInputsForOp().add(e.getKey());
        }

        Preconditions.checkState(remainingNodes.isEmpty(), "%s Unprocessed nodes: %s", remainingNodes.size(), remainingNodes.keySet());

        return sd;
    }


    /**
     * Get the shape from a TensorShapeProto
     *
     * @param tensorShapeProto Shape
     * @return Shape as long[]
     */
    private static long[] shapeFromShapeProto(TensorShapeProto tensorShapeProto) {
        long[] shape = new long[tensorShapeProto.getDimList().size()];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = tensorShapeProto.getDim(i).getSize();
        }

        return shape;
    }

    /**
     * Convert from TF proto datatype to ND4J datatype
     *
     * @param tfType TF datatype
     * @return ND4J datatype
     */
    public static org.nd4j.linalg.api.buffer.DataType convertType(org.tensorflow.framework.DataType tfType) {
        switch (tfType) {
            case DT_DOUBLE:
                return org.nd4j.linalg.api.buffer.DataType.DOUBLE;
            case DT_FLOAT:
                return org.nd4j.linalg.api.buffer.DataType.FLOAT;
            case DT_HALF:
                return org.nd4j.linalg.api.buffer.DataType.HALF;
            case DT_BFLOAT16:
                return org.nd4j.linalg.api.buffer.DataType.BFLOAT16;
            case DT_INT8:
                return org.nd4j.linalg.api.buffer.DataType.BYTE;
            case DT_INT16:
                return org.nd4j.linalg.api.buffer.DataType.SHORT;
            case DT_INT32:
                return org.nd4j.linalg.api.buffer.DataType.INT;
            case DT_INT64:
                return org.nd4j.linalg.api.buffer.DataType.LONG;
            case DT_UINT8:
                return org.nd4j.linalg.api.buffer.DataType.UBYTE;
            case DT_STRING:
                return org.nd4j.linalg.api.buffer.DataType.UTF8;
            case DT_BOOL:
                return org.nd4j.linalg.api.buffer.DataType.BOOL;

            default:
                return org.nd4j.linalg.api.buffer.DataType.UNKNOWN;
        }
    }

    /**
     * @return True if the specified name represents a control dependency (starts with "^")
     */
    protected static boolean isControlDep(String name) {
        return name.startsWith("^");
    }

    /**
     * @return The specified name without the leading "^" character (if any) that appears for control dependencies
     */
    protected static String stripControl(String name) {
        if (name.startsWith("^")) {
            return name.substring(1);
        }
        return name;
    }

    /**
     * Remove the ":1" etc suffix for a variable name to get the op name
     *
     * @param varName Variable name
     * @return Variable name without any number suffix
     */
    protected static String stripVarSuffix(String varName) {
        if (varName.matches(".*:\\d+")) {
            int idx = varName.lastIndexOf(':');
            String ret = varName.substring(0, idx);
            return ret;
        }
        return varName;
    }

    /**
     * Convert the tensor to an NDArray (if possible and if array is available)
     *
     * @param node Node to get NDArray from
     * @return NDArray
     */
    public static INDArray getNDArrayFromTensor(NodeDef node) {
        //placeholder of some kind
        if (!node.getAttrMap().containsKey("value")) {
            return null;
        }

        val tfTensor = node.getAttrOrThrow("value").getTensor();
        INDArray out = mapTensorProto(tfTensor);
        return out;
    }

    /**
     * Convert a TensorProto to an INDArray
     *
     * @param tfTensor Tensor proto
     * @return INDArray
     */
    public static INDArray mapTensorProto(TensorProto tfTensor) {
        TFTensorMapper m = TFTensorMappers.newMapper(tfTensor);
        if (m == null) {
            throw new RuntimeException("Not implemented datatype: " + tfTensor.getDtype());
        }
        INDArray out = m.toNDArray();
        return out;
    }

    @Deprecated //To be removed
    public static NodeDef getNodeWithNameFromGraph(GraphDef graph, String name) {
        for (int i = 0; i < graph.getNodeCount(); i++) {
            val node = graph.getNode(i);
            if (node.getName().equals(name))
                return node;
        }

        return null;
    }

    @Deprecated //To be removed
    public static INDArray getArrayFrom(NodeDef nodeDef, GraphDef graph) {
        if (nodeDef == null) {
            return null;
        }

        return getNDArrayFromTensor(nodeDef);
    }

    /**
     * Init a function's attributes
     *
     * @param mappedTfName      the tensorflow name to pick (sometimes ops have multiple names
     * @param on                the function to map
     * @param attributesForNode the attributes for the node
     * @param node
     * @param graph
     * @deprecated To be removed
     */
    @Deprecated
    public static void initFunctionFromProperties(String mappedTfName, DifferentialFunction on, Map<String, AttrValue> attributesForNode, NodeDef node, GraphDef graph) {
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
        Map<String, PropertyMapping> map;
        if (attributeAdapters == null || !attributeAdapters.containsKey(mappedTfName)) {
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

        for (Map.Entry<String, PropertyMapping> entry : map.entrySet()) {
            val tfAttrName = entry.getValue().getTfAttrName();
            val currentField = fields.get(entry.getKey());

            AttributeAdapter adapter = null;
            if (attributeAdapters != null && !attributeAdapters.isEmpty()) {
                val mappers = attributeAdapters.get(mappedTfName);
                val adapterFor = mappers.get(entry.getKey());
                adapter = adapterFor;
            }


            if (tfAttrName != null) {
                if (currentField == null) {
                    continue;
                }

                if (attributesForNode.containsKey(tfAttrName)) {
                    val attr = attributesForNode.get(tfAttrName);
                    switch (attr.getValueCase()) {
                        case B:
                            if (adapter != null) {
                                adapter.mapAttributeFor(attr.getB(), currentField, on);
                            }
                            break;
                        case F:
                            break;
                        case FUNC:
                            break;
                        case S:
                            val setString = attr.getS().toStringUtf8();
                            if (adapter != null) {
                                adapter.mapAttributeFor(setString, currentField, on);
                            } else
                                on.setValueFor(currentField, setString);
                            break;
                        case I:
                            val setInt = (int) attr.getI();
                            if (adapter != null) {
                                adapter.mapAttributeFor(setInt, currentField, on);
                            } else
                                on.setValueFor(currentField, setInt);
                            break;
                        case SHAPE:
                            val shape = attr.getShape().getDimList();
                            int[] dimsToSet = new int[shape.size()];
                            for (int i = 0; i < dimsToSet.length; i++) {
                                dimsToSet[i] = (int) shape.get(i).getSize();
                            }

                            if (adapter != null) {
                                adapter.mapAttributeFor(dimsToSet, currentField, on);
                            } else
                                on.setValueFor(currentField, dimsToSet);
                            break;
                        case VALUE_NOT_SET:
                            break;
                        case PLACEHOLDER:
                            break;
                        case LIST:
                            val setList = attr.getList();
                            if (!setList.getIList().isEmpty()) {
                                val intList = Ints.toArray(setList.getIList());
                                if (adapter != null) {
                                    adapter.mapAttributeFor(intList, currentField, on);
                                } else
                                    on.setValueFor(currentField, intList);
                            } else if (!setList.getBList().isEmpty()) {
                                break;
                            } else if (!setList.getFList().isEmpty()) {
                                val floats = Floats.toArray(setList.getFList());
                                if (adapter != null) {
                                    adapter.mapAttributeFor(floats, currentField, on);
                                } else
                                    on.setValueFor(currentField, floats);
                                break;
                            } else if (!setList.getFuncList().isEmpty()) {
                                break;
                            } else if (!setList.getTensorList().isEmpty()) {
                                break;
                            }
                            break;
                        case TENSOR:
                            val tensorToGet = TFGraphMapper.mapTensorProto(attr.getTensor());
                            if (adapter != null) {
                                adapter.mapAttributeFor(tensorToGet, currentField, on);
                            } else
                                on.setValueFor(currentField, tensorToGet);
                            break;
                        case TYPE:
                            if (adapter != null) {
                                adapter.mapAttributeFor(attr.getType(), currentField, on);
                            }
                            break;
                    }
                }
            } else if (entry.getValue().getTfInputPosition() != null) {


                int position = entry.getValue().getTfInputPosition();
                if (position < 0) {
                    position += node.getInputCount();
                }

                val inputFromNode = TFGraphMapper.getNodeWithNameFromGraph(graph, node.getInput(position));
                INDArray tensor = inputFromNode != null ? TFGraphMapper.getNDArrayFromTensor(inputFromNode) : null;
                if (tensor == null) {
                    tensor = on.getSameDiff().getArrForVarName(getNodeName(node.getInput(position)));
                }


                if (tensor != null) {
                    //use adapter instead of direct mapping just like above
                    if (adapter != null) {
                        adapter.mapAttributeFor(tensor, currentField, on);
                    } else {
                        if (currentField.getType().equals(int[].class)) {
                            on.setValueFor(currentField, tensor.data().asInt());
                        } else if (currentField.getType().equals(double[].class)) {
                            on.setValueFor(currentField, tensor.data().asDouble());

                        } else if (currentField.getType().equals(float[].class)) {
                            on.setValueFor(currentField, tensor.data().asFloat());

                        } else if (currentField.getType().equals(INDArray.class)) {
                            on.setValueFor(currentField, tensor);
                        } else if (currentField.getType().equals(int.class)) {
                            on.setValueFor(currentField, tensor.getInt(0));
                        } else if (currentField.getType().equals(double.class)) {
                            on.setValueFor(currentField, tensor.getDouble(0));
                        } else if (currentField.getType().equals(float.class)) {
                            on.setValueFor(currentField, tensor.getFloat(0));
                        }
                    }
                }
            }
        }
    }

    /**
     * Map a tensorflow node name
     * to the samediff equivalent
     * for import
     *
     * @param name the name to change
     * @return the input tensorflow name
     * @deprecated To be removed
     */
    @Deprecated
    public static String getNodeName(String name) {
        //tensorflow adds colons to the end of variables representing input index, this strips those off
        String ret = name;
        if (ret.startsWith("^"))
            ret = ret.substring(1);
        if (ret.endsWith("/read")) {
            ret = ret.replace("/read", "");
        }
        if (ret.endsWith(":0")) {
            ret = ret.substring(0, ret.length() - 2);
        }
        return ret;
    }

    /**
     * Determine if the node represents a variable node (based on op name)
     *
     * @param nodeDef Node to check if a variable
     * @return True if a variable node
     */
    public static boolean isVariableNode(NodeDef nodeDef) {
        boolean isVar = nodeDef.getOp().startsWith("VariableV") || nodeDef.getOp().equalsIgnoreCase("const");
        return isVar;
    }

    /**
     * Determine if the node is a placeholder
     *
     * @param nodeDef Node to check
     * @return True if the node is a placeholder
     */
    public static boolean isPlaceHolder(NodeDef nodeDef) {
        return nodeDef.getOp().startsWith("Placeholder");
    }
}
