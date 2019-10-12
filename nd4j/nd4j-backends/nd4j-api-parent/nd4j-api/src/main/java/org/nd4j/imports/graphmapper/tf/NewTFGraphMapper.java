package org.nd4j.imports.graphmapper.tf;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMapper;
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMappers;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.guava.base.Preconditions;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;
import org.tensorflow.framework.TensorProto;

import java.io.*;
import java.util.*;

@Slf4j
public class NewTFGraphMapper {

    public static SameDiff importGraph(@NonNull File f){
        Preconditions.checkState(f.exists(), "File does not exist: %s", f);


        GraphDef tfGraph;

        try(InputStream is = new BufferedInputStream(new FileInputStream(f))){
            tfGraph = GraphDef.parseFrom(is);
        } catch (IOException e){
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

        Map<String, Set<String>> nodeInputTo = new HashMap<>();     // For op x -> y, x is key, y is value

        for( int i=0; i<nNodes; i++ ){
            NodeDef nd = tfGraph.getNode(i);
            String op = nd.getOp();
            String name = nd.getName();

            int nInputs = nd.getInputCount();

            if("Const".equals(op) || nInputs == 0){
                availableToAdd.add(nd);
                availableToAddSet.add(name);
            } else {
                remainingNodes.put(name, nd);
                for( int in=0; in<nInputs; in++ ){
                    String inName = nd.getInput(in);
                    if(!nodeInputTo.containsKey(inName)){
                        nodeInputTo.put(inName, new HashSet<String>());
                    }
                    nodeInputTo.get(inName).add(name);
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

            if("Const".equals(opName)){
                //Get array, create a constant
                TensorProto tfTensor = nd.getAttrOrThrow("value").getTensor();
                TFTensorMapper m = TFTensorMappers.newMapper(tfTensor);
                INDArray arr = m.toNDArray();
                sd.constant(name, arr);
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
                for(int i=0; i<nIn; i++){
                    //TODO handle control dependencies
                    String inName = nd.getInput(i);
                    inNames.add(inName);

                    //Update Variable.inputsForOp for all variables that feed into this op
                    // Such variables must have already been created, given we process in order
                    Variable v = sd.getVariables().get(inName);
                    if(v.getInputsForOp() == null)
                        v.setInputsForOp(new ArrayList<String>());
                    Preconditions.checkState(!v.getInputsForOp().contains(name), "Variable %s already an input for %s", inName, name);
                    v.getInputsForOp().add(name);
                }

                //Create SameDiffOp instance and add to graph
                SameDiffOp op = SameDiffOp.builder()
                        .name(name)
                        .op(df)
                        .inputsToOp(inNames)
                        //.outputsOfOp(outNames)    //We'll set this later
                        .controlDeps(null)              //TODO
                        .build();
                sd.getOps().put(name, op);


                Map<String, AttrValue> attrMap = nd.getAttrMap();
                df.initFromTensorFlow(nd, sd, attrMap, null);

                //DType calculate for output variables (set/correct if necessary)
                List<String> newInNames = sd.getOps().get(name).getInputsToOp();        //Just in case import has modified this, like for concat case
                List<DataType> newInDtypes = new ArrayList<>(newInNames.size());
                for( String s : newInNames ){
                    newInDtypes.add(sd.getVariable(s).dataType());
                }

//                List<DataType> outDTypes = df.calculateOutputDataTypes(inputDTypes);
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
                        String inName = nextOpDef.getInput(i);
                        if(!sd.hasVariable(inName)){
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

}
