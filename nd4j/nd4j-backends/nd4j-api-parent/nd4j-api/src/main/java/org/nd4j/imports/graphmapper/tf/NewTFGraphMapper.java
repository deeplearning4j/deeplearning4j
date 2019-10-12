package org.nd4j.imports.graphmapper.tf;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMapper;
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMappers;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.tensorflow.framework.*;

import java.io.*;
import java.util.*;

@Slf4j
public class NewTFGraphMapper {

    public static SameDiff importGraph(@NonNull File f) {
        Preconditions.checkState(f.exists(), "File does not exist: %s", f);
        try (InputStream is = new BufferedInputStream(new FileInputStream(f))) {
            return importGraph(is);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static SameDiff importGraph(@NonNull InputStream is) {
        GraphDef tfGraph;
        try {
            tfGraph = GraphDef.parseFrom(is);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        int nNodes = tfGraph.getNodeCount();
        for( int i=0; i<nNodes; i++ ){
            NodeDef nd = tfGraph.getNode(i);
            String op = nd.getOp();
            String name = nd.getName();
            System.out.println(i + " - op=" + op + ", name=\"" + name + "\"");
        }

        /*
        First, build an in-memory representation of the graph that allows us to build the graph incrementally
        If we can build the graph incrementally, we can make sure that the added variables are set up with the correct
        datatype and (once implemented) greedy shape inference
         */
        Set<String> availableToAddSet = new HashSet<>();            //TODO maybe unnecessary?
        Queue<NodeDef> availableToAdd = new LinkedList<>();

        Map<String, NodeDef> remainingNodes = new HashMap<>();          //All other nodes, not in availableToAdd

        Map<String, Set<String>> nodeInputTo = new HashMap<>();     // For op x -> y, x is key, y is value. Note that these are OP names not VARIABLE names

        for( int i=0; i<nNodes; i++ ){
            NodeDef nd = tfGraph.getNode(i);
            String op = nd.getOp();
            String name = nd.getName();

            int nInputs = nd.getInputCount();

            if("Const".equals(op) || "Placeholder".equals(op) || nInputs == 0) {
                availableToAdd.add(nd);
                availableToAddSet.add(name);
            } else {
                remainingNodes.put(name, nd);
                for( int in=0; in<nInputs; in++ ){
                    String inOpName = stripControl(nd.getInput(in));
                    inOpName = stripVarSuffix(inOpName);
//                    if(isControlDep(inName)){
//                        //Ignore for import ordering. Relevant for execution, however
//                        continue;
//                    }

                    if(!nodeInputTo.containsKey(inOpName)){
                        nodeInputTo.put(inOpName, new HashSet<String>());
                    }
                    nodeInputTo.get(inOpName).add(name);
                }
            }
        }

        //Go through ops in order, and add to the graph
        SameDiff sd = SameDiff.create();
        while(!availableToAdd.isEmpty()){
            NodeDef nd = availableToAdd.remove();
            String name = nd.getName();
            String opName = nd.getOp();
            int nIn = nd.getInputCount();
            log.info("Adding operation to graph: {} (name={})", opName, name);

            availableToAddSet.remove(name);

            if("Const".equals(opName)) {
                //Get array, create a constant
                TensorProto tfTensor = nd.getAttrOrThrow("value").getTensor();
                TFTensorMapper m = TFTensorMappers.newMapper(tfTensor);
                INDArray arr = m.toNDArray();
                sd.constant(name, arr);
            } else if("Placeholder".equals(opName)){

                Map<String,AttrValue> attrMap = nd.getAttrMap();
                TensorShapeProto shapeProto = attrMap.get("shape").getShape();
                long[] shape = shapeFromShapeProto(shapeProto);

                org.tensorflow.framework.DataType tfDtype = attrMap.get("dtype").getType();
                DataType dt = convertType(tfDtype);
                sd.placeHolder(name, dt, shape);
            } else {
                /*
                Normal ops. Process in the following order:
                1. Create the op instance
                2. Add op to graph
                3. Import from TF (to set attributes)
                4. Calculate output dtypes
                5. Create and add output variables to graph

                Note: one constraint on this order is that some op's import modify the graph structure.
                Notable example: concat op - it removes the axis op and converts the value to an iArg
                https://github.com/eclipse/deeplearning4j/issues/8285
                 */
                DifferentialFunction dfInstance = DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(opName);
                Preconditions.checkState(dfInstance != null, "Could not find class for TF Ops: {}", opName);

                DifferentialFunction df;
                try {
                    df = dfInstance.getClass().newInstance();
                } catch (Throwable t){
                    //Should never happen because function was already created via no-arg constructor earlier
                    throw new RuntimeException(t);
                }
                df.setSameDiff(sd);
                df.setOwnName(name);

                //Process inputs
                List<String> inNames = new ArrayList<>(nIn);
                List<String> controlDeps = null;
                for(int i=0; i<nIn; i++){
                    //TODO handle control dependencies
                    String origInName = nd.getInput(i);
                    String inName = stripControl(origInName);
                    boolean isControlDep = isControlDep(origInName);
                    if(isControlDep){
                        if(controlDeps == null)
                            controlDeps = new ArrayList<>();
                        controlDeps.add(inName);
                    }

                    if(!isControlDep) {
                        inNames.add(inName);
                    }

                    //Update Variable.inputsForOp for all variables that feed into this op
                    // Such variables must have already been created, given we process in order
                    Variable v = sd.getVariables().get(inName);

                    if(!isControlDep && (v.getInputsForOp() == null || !v.getInputsForOp().contains(name))){
                        //May already be present - for example, add(x,x)
                        if(v.getInputsForOp() == null)
                            v.setInputsForOp(new ArrayList<String>());
                        v.getInputsForOp().add(name);
                    } else if(isControlDep){
                        if(v.getControlDepsForOp() == null)
                            v.setControlDepsForOp(new ArrayList<String>());
                        if(!v.getControlDepsForOp().contains(name)){
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
                List<DataType> newInDtypes = new ArrayList<>(newInNames.size());
                for( String s : newInNames ){
                    newInDtypes.add(sd.getVariable(s).dataType());
                }

                List<DataType> outDTypes = df.calculateOutputDataTypes(newInDtypes);
                SDVariable[] outSDVars = new SDVariable[outDTypes.size()];
                Variable[] outVars = new Variable[outDTypes.size()];
                List<String> outNames = new ArrayList<>(outDTypes.size());

                //Create output variables and add to graph
                for( int i=0; i<outDTypes.size(); i++ ){
                    DataType dt = outDTypes.get(i);
                    String varName = name + (i == 0 ? "" : ":" + i);
                    outSDVars[i] = sd.var(varName, VariableType.ARRAY, null, dt, (long[])null);
                    outNames.add(varName);

                    outVars[i] = Variable.builder()
                            .name(varName)
                            .variable(outSDVars[i])
                            .inputsForOp(null)          //This is updated incrementally as other ops are added
                            .controlDepsForOp(null)     //TODO
                            .controlDepsForVar(null)    //TODO
                            .outputOfOp(name)
                            .outputOfOpIdx(i)
                            .build();

                    sd.getVariables().put(varName, outVars[i]);
                    log.info("Added variable to graph: {} (output of op {})", varName, name);
                }
                sd.getOps().get(name).setOutputsOfOp(outNames);

                log.info("Imported op: {} (name={})", opName, name);
            }


            //Now that we have just added an op (or variable) - check what this feeds into, and see what we can now process
            // as a result
            if(nodeInputTo.containsKey(name)){
                Set<String> set = nodeInputTo.get(name);
                for(String nextOp : set){
                    NodeDef nextOpDef = remainingNodes.get(nextOp);
                    int nInNext = nextOpDef.getInputCount();
                    boolean allAlreadyInGraph = true;
                    for(int i=0; i<nInNext; i++ ){
                        String s = nextOpDef.getInput(i);
//                        if(isControlDep(s)){
//                            continue;
//                        }

                        String inName = stripControl(nextOpDef.getInput(i));

                        log.info("Input: {}, {}", s, inName);

                        if(!sd.hasVariable(inName)){
                            log.info("Not found: {} for op {}", inName, nextOpDef.getName());
                            allAlreadyInGraph = false;
                            break;
                        }
                    }

                    if(allAlreadyInGraph){
                        //Can process this op, add it to the queue for processing
                        remainingNodes.remove(nextOp);
                        availableToAdd.add(nextOpDef);
                        availableToAddSet.add(nextOp);
                        log.info("Added to processing queue: {} (name={})", nextOpDef.getOp(), nextOp);
                    }
                }
            }

            //Finally, remove the just processed op from remainingNodes map:
            remainingNodes.remove(name);
        }

        Preconditions.checkState(remainingNodes.isEmpty(), "Unprocessed nodes: %s", remainingNodes.keySet());

        return sd;
    }


    private static long[] shapeFromShapeProto(TensorShapeProto tensorShapeProto) {
        long[] shape = new long[tensorShapeProto.getDimList().size()];
        for(int i = 0; i < shape.length; i++) {
            shape[i] =  tensorShapeProto.getDim(i).getSize();
        }

        return shape;
    }

    public static org.nd4j.linalg.api.buffer.DataType convertType(org.tensorflow.framework.DataType tfType){
        switch(tfType) {
            case DT_DOUBLE: return org.nd4j.linalg.api.buffer.DataType.DOUBLE;
            case DT_FLOAT: return org.nd4j.linalg.api.buffer.DataType.FLOAT;
            case DT_HALF: return org.nd4j.linalg.api.buffer.DataType.HALF;
            case DT_BFLOAT16: return org.nd4j.linalg.api.buffer.DataType.BFLOAT16;
            case DT_INT8: return org.nd4j.linalg.api.buffer.DataType.BYTE;
            case DT_INT16: return org.nd4j.linalg.api.buffer.DataType.SHORT;
            case DT_INT32: return org.nd4j.linalg.api.buffer.DataType.INT;
            case DT_INT64: return org.nd4j.linalg.api.buffer.DataType.LONG;
            case DT_UINT8: return org.nd4j.linalg.api.buffer.DataType.UBYTE;
            case DT_STRING: return org.nd4j.linalg.api.buffer.DataType.UTF8;
            case DT_BOOL: return org.nd4j.linalg.api.buffer.DataType.BOOL;

            default: return org.nd4j.linalg.api.buffer.DataType.UNKNOWN;
        }
    }

    protected static boolean isControlDep(String name){
        return name.startsWith("^");
    }

    protected static String stripControl(String name){
        if(name.startsWith("^")){
            return name.substring(1);
        }
        return name;
    }

    /**
     * Remove the ":1" etc suffix for a variable name to get the op name
     * @param varName
     * @return
     */
    protected static String stripVarSuffix(String varName){
        if(varName.matches(".*:\\d+")){
            int idx = varName.lastIndexOf(':');
            String ret = varName.substring(0, idx);
            return ret;
        }
        return varName;
    }

}
