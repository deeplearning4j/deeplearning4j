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

package org.nd4j.imports.graphmapper;

import com.github.os72.protobuf351.Message;
import com.github.os72.protobuf351.TextFormat;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.IOUtils;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.io.*;
import java.util.*;

/**
 * Base implementation for importing a graph
 *
 * @param <GRAPH_TYPE> the type of graph
 * @param <NODE_TYPE>  the type of node
 * @param <ATTR_TYPE>  the attribute type
 */
@Slf4j
public abstract class BaseGraphMapper<GRAPH_TYPE, NODE_TYPE, ATTR_TYPE, TENSOR_TYPE> implements GraphMapper<GRAPH_TYPE, NODE_TYPE, ATTR_TYPE, TENSOR_TYPE> {


    @Override
    public Op.Type opTypeForNode(NODE_TYPE nodeDef) {
        DifferentialFunction opWithTensorflowName = getMappedOp(getOpType(nodeDef));
        if (opWithTensorflowName == null)
            throw new NoOpNameFoundException("No op found with name " + getOpType(nodeDef));
        Op.Type type = opWithTensorflowName.opType();
        return type;

    }


    @Override
    public void mapProperties(DifferentialFunction on, NODE_TYPE node, GRAPH_TYPE graph, SameDiff sameDiff, Map<String, Map<String, PropertyMapping>> propertyMappings) {
        val mappings = propertyMappings.get(getOpType(node));
        if (mappings == null || mappings.isEmpty()) {
            return;
        }


        for (val entry : mappings.entrySet()) {
            mapProperty(entry.getKey(), on, node, graph, sameDiff, propertyMappings);
        }
    }


    /**
     * @param inputStream
     * @return
     */
    @Override
    public SameDiff importGraph(InputStream inputStream) {
        return importGraph(inputStream, Collections.<String, OpImportOverride<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE>>emptyMap(), null);
    }

    @Override
    public SameDiff importGraph(InputStream inputStream, Map<String,? extends OpImportOverride<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE>> opImportOverrides,
                                OpImportFilter<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE> opFilter) {
        GRAPH_TYPE def = readGraph(inputStream, opImportOverrides);
        return importGraph(def, opImportOverrides, opFilter);
    }

    protected GRAPH_TYPE readGraph(InputStream inputStream, Map<String,? extends OpImportOverride<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE>> opImportOverrides) {
        byte[] bytes = null;
        GRAPH_TYPE def = null;
        try {
            bytes = IOUtils.toByteArray(inputStream);   //Buffers internally
            def = parseGraphFrom(bytes);
        } catch (IOException e) {
            try (BufferedInputStream bis2 = new BufferedInputStream(new ByteArrayInputStream(bytes)); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
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

        return def;
    }

    /**
     * @param graphFile
     * @return
     */
    @Override
    public SameDiff importGraph(File graphFile) {
        return importGraph(graphFile, Collections.<String, OpImportOverride<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE>>emptyMap(), null);
    }

    @Override
    public SameDiff importGraph(File graphFile, Map<String,? extends OpImportOverride<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE>> opImportOverrides,
                                OpImportFilter<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE> opFilter) {
        GRAPH_TYPE def = null;
        try (FileInputStream fis = new FileInputStream(graphFile)) {
            return importGraph(fis, opImportOverrides, opFilter);
        } catch (Exception e) {
            throw new ND4JIllegalStateException("Error encountered loading graph file: " + graphFile.getAbsolutePath(), e);
        }
    }

    @Override
    public Map<String, NODE_TYPE> nameIndexForGraph(GRAPH_TYPE graph) {
        List<NODE_TYPE> nodes = getNodeList(graph);
        Map<String, NODE_TYPE> ret = new HashMap<>();
        for (NODE_TYPE node : nodes) {
            ret.put(getName(node), node);
        }
        return ret;
    }

    @Override
    public Map<String, NODE_TYPE> nodesByName(GRAPH_TYPE graph) {
        val nodeTypes = getNodeList(graph);
        Map<String, NODE_TYPE> ret = new LinkedHashMap<>();
        for (int i = 0; i < nodeTypes.size(); i++) {
            ret.put(getName(nodeTypes.get(i)), nodeTypes.get(i));
        }
        return ret;
    }

    /**
     * This method converts given TF
     *
     * @param tfGraph
     * @return
     */
    @Override
    public SameDiff importGraph(GRAPH_TYPE tfGraph) {
        return importGraph(tfGraph, Collections.<String, OpImportOverride<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE>>emptyMap(), null);
    }

    @Override
    public SameDiff importGraph(GRAPH_TYPE tfGraph, Map<String,? extends OpImportOverride<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE>> opImportOverrides,
                                OpImportFilter<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE> opFilter) {

        SameDiff diff = SameDiff.create();
        ImportState<GRAPH_TYPE, TENSOR_TYPE> importState = new ImportState<>();
        importState.setSameDiff(diff);
        importState.setGraph(tfGraph);

        Map<String,TENSOR_TYPE> variablesForGraph = variablesForGraph(tfGraph);
        importState.setVariables(variablesForGraph);


        //Add each of the variables first - before importing ops
        Map<String, Boolean> stringNodes = new HashMap<>();      //Key: name of string variable. Value: if it's a constant
        for (Map.Entry<String, TENSOR_TYPE> entry : variablesForGraph.entrySet()) {
            if (shouldSkip((NODE_TYPE) entry.getValue())) {    //TODO only works for TF
                //Skip some nodes, for example reduction indices (a lot of ND4J/SameDiff ops use int[] etc, not an INDArray/Variable)
                continue;
            }

            //First: check if we're skipping the op entirely. If so: don't create the output variables for it.
            NODE_TYPE node = (NODE_TYPE) entry.getValue();      //TODO this only works for TF
            String opType = getOpType(node);
            String opName = getName(node);
            if(opFilter != null && opFilter.skipOp(node, importState.getSameDiff(), null, importState.getGraph() )){
                log.info("Skipping variables for op: {} (name: {})", opType, opName);
                continue;
            }

            //Similarly, if an OpImportOverride is defined, don't create the variables now, as these might be the wrong type
            //For example, the OpImportOverride might replace the op with some placeholders
            // If we simply created them now, we'd create the wrong type (Array not placeholder)
            if(opImportOverrides != null && opImportOverrides.containsKey(opType)){
                log.info("Skipping variables for op due to presence of OpImportOverride: {} (name: {})", opType, opName);
                continue;
            }


            DataType dt = dataTypeForTensor(entry.getValue(), 0);
            INDArray arr = getNDArrayFromTensor(entry.getKey(), entry.getValue(), tfGraph);
            long[] shape = hasShape((NODE_TYPE) entry.getValue()) ? getShape((NODE_TYPE) entry.getValue()) : null;   //TODO only works for TF

            //Not all variables have datatypes available on import - we have to infer these at a later point
            // so we'll leave datatypes as null and infer them once all variables/ops have been imported
            if(dt == DataType.UNKNOWN)
                dt = null;

            if (isPlaceHolder(entry.getValue())) {
                diff.placeHolder(entry.getKey(), dt, shape);
            } else if (isConstant(entry.getValue())) {
                Preconditions.checkNotNull(arr, "Array is null for placeholder variable %s", entry.getKey());
                diff.constant(entry.getKey(), arr);
            } else {
                //Could be variable, or could be array type (i.e., output of op/"activations")
                //TODO work out which!

                SDVariable v;
                if(shape == null){
                    //No shape -> probably not a variable...
                    v = diff.var(entry.getKey(), VariableType.ARRAY, null, dt, (long[])null);
                } else {
                    v = diff.var(entry.getKey(), dt, shape);
                }
                if (arr != null)
                    diff.associateArrayWithVariable(arr, v);
            }

//            NODE_TYPE node = (NODE_TYPE) entry.getValue();      //TODO this only works for TF
            List<String> controlDependencies = getControlDependencies(node);
            if (controlDependencies != null) {
                Variable v = diff.getVariables().get(entry.getKey());
                v.setControlDeps(controlDependencies);
            }
        }

        //Map ops
        val tfNodesList = getNodeList(tfGraph);
        for (NODE_TYPE node : tfNodesList) {
            String opType = getOpType(node);
            OpImportOverride<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE> importOverride = null;
            if(opImportOverrides != null){
                importOverride = opImportOverrides.get(opType);
            }

            if(opFilter != null && opFilter.skipOp(node, importState.getSameDiff(), null, null)){
                String opName = getName(node);
                log.info("Skipping op due to op filter: {}", opType, opName);
                continue;
            }

            if (!opsToIgnore().contains(opType) || isOpIgnoreException(node)) {
                mapNodeType(node, importState, importOverride, opFilter);
            }
        }


        /*
        At this point, we have a few remaining things to do:
        1. Make sure all datatypes are set on all variables. TF doesn't have datatype info an all op outputs for some reason, so we have to infer in manually
        2. Make sure all op output variables have been created
        3. Make sure all SameDiffOp.outputsOfOp is set
        4. Make sure all Variable.outputOfOp is set
        5. Make sure all Variable.controlDepsForVar have been populated (reverse lookup of Variable.controlDeps)
         */

        //Make sure Variable.outputOfOp is set
        for(Variable v : diff.getVariables().values()){
            if(v.getVariable().isPlaceHolder() || v.getVariable().isConstant())
                continue;

            //Expect variable names of output variables to be: opName, opName:1, opName:2, etc
            String n = v.getName();
            String opName = n;
            if(v.getName().matches(".*:\\d+")){
                //i.e., "something:2"
                int idx = n.lastIndexOf(':');
                opName = n.substring(0,idx);
            }

            if(diff.getOps().containsKey(opName)) {
                //Variable is the output of an op
                v.setOutputOfOp(opName);

                //Also double check variable type...
                if(v.getVariable().getVariableType() != VariableType.ARRAY)
                    v.getVariable().setVariableType(VariableType.ARRAY);
            }
        }

        //Initialize any missing output variables
        for (SameDiffOp op : diff.getOps().values()) {
            DifferentialFunction df = op.getOp();
            initOutputVariables(diff, df);
        }

        //Make sure all Variable.controlDepsForVar have been populated (reverse lookup of Variable.controlDeps)
        //i.e., if control dependency x -> y exists, then:
        // (a) x.controlDepsForVar should contain "y"
        // (b) y.controlDeps should contain "x"
        //Need to do this before output datatype calculation, as these control dep info is used in sessions
        for(Map.Entry<String,Variable> e : diff.getVariables().entrySet()){
            Variable v = e.getValue();
            if(v.getControlDeps() != null){
                for(String s : v.getControlDeps()){
                    Variable v2 = diff.getVariables().get(s);
                    if(v2.getControlDepsForVar() == null)
                        v2.setControlDepsForVar(new ArrayList<String>());
                    if(!v2.getControlDepsForVar().contains(e.getKey())){
                        //Control dep v2 -> v exists, so put v.name into v2.controlDepsForVar
                        v2.getControlDepsForVar().add(e.getKey());
                    }
                }
            }
        }

        //Same thing for op control dependencies...
        for(Map.Entry<String,SameDiffOp> e : diff.getOps().entrySet()){
            SameDiffOp op = e.getValue();
            if(op.getControlDeps() != null){
                for(String s : op.getControlDeps()){
                    //Control dependency varS -> op exists
                    Variable v = diff.getVariables().get(s);
                    if(v.getControlDepsForOp() == null)
                        v.setControlDepsForOp(new ArrayList<String>());
                    if(!v.getControlDepsForOp().contains(e.getKey()))
                        v.getControlDepsForOp().add(e.getKey());
                }
            }
        }


        //Infer variable datatypes to ensure all variables have datatypes...
        boolean anyUnknown = false;
        for(SDVariable v : diff.variables()){
            if(v.dataType() == null)
                anyUnknown = true;
        }
        if(anyUnknown){
            Map<String,DataType> dataTypes = diff.calculateOutputDataTypes();
            for(SDVariable v : diff.variables()){
                if(v.dataType() == null){
                    v.setDataType(dataTypes.get(v.getVarName()));
                }
            }
        }

        //Validate the graph structure
        validateGraphStructure(diff);

        return diff;
    }

    protected void initOutputVariables(SameDiff sd, DifferentialFunction df) {
        String[] outNames = sd.getOutputsForOp(df);
        SDVariable[] outVars;
        if (outNames == null) {
            outVars = sd.generateOutputVariableForOp(df, df.getOwnName() != null ? df.getOwnName() : df.opName(), true);
            outNames = new String[outVars.length];
            for (int i = 0; i < outVars.length; i++) {
                outNames[i] = outVars[i].getVarName();
            }
            sd.getOps().get(df.getOwnName()).setOutputsOfOp(Arrays.asList(outNames));
        }

        for (String s : outNames) {
            sd.getVariables().get(s).setOutputOfOp(df.getOwnName());
        }
    }


    @Override
    public boolean validTensorDataType(TENSOR_TYPE tensorType) {
        return dataTypeForTensor(tensorType, 0) != DataType.UNKNOWN;
    }

    public void validateGraphStructure(SameDiff sameDiff) {
        //First: Check placeholders. When SDVariables are added with null shapes, these can be interpreted as a placeholder
        // but null shapes might simply mean shape isn't available during import right when the variable is added
        //Idea here: if a "placeholder" is the output of any function, it's not really a placeholder
        for (SDVariable v : sameDiff.variables()) {
            String name = v.getVarName();
            if (sameDiff.isPlaceHolder(name)) {
                String varOutputOf = sameDiff.getVariables().get(name).getOutputOfOp();
            }
        }

        //Second: check that all op inputs actually exist in the graph
        for (SameDiffOp op : sameDiff.getOps().values()) {
            List<String> inputs = op.getInputsToOp();
            if (inputs == null)
                continue;

            for (String s : inputs) {
                if (sameDiff.getVariable(s) == null) {
                    throw new IllegalStateException("Import validation failed: op \"" + op.getName() + "\" of type " + op.getOp().getClass().getSimpleName()
                            + " has input \"" + s + "\" that does not have a corresponding variable in the graph");
                }
            }
        }
    }

}
