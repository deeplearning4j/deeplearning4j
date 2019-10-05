/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.internal;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.memory.SimpleSessionMemoryMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.controlflow.If;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 * InferenceSession: Performs inference (forward pass) on a SameDiff instance to get the outputs of the requested nodes.
 * Dynamically (in AbstractSession) calculates the required subgraph to execute to get the required outputs.
 *
 * @author Alex Black
 */
@Slf4j
public class InferenceSession2 extends AbstractSession<INDArray,SameDiffOp> {
    private static final String SCOPE_PANIC_MSG = "If required, arrays in workspaces can be detached using INDArray.detach() before being passed to the SameDiff instance.\n" +
            "Alternatively, arrays defined in a workspace must be replaced after the workspace has been closed.";

    private SessionMemMrg mmgr;

    public InferenceSession2(@NonNull SameDiff sameDiff) {
        super(sameDiff);

        mmgr = new SimpleSessionMemoryMgr();
    }

    @Override
    protected Map<String,INDArray> preprocessPlaceholders(Map<String,INDArray> placeholders){
        //Handle casting of the input array automatically.
        //The idea here is to avoid unexpected errors if the user (for example) tries to perform inference with a double
        // array for a float placeholder
        if(placeholders == null || placeholders.isEmpty()){
            return placeholders;
        }

        Map<String,INDArray> out = new HashMap<>();
        for(Map.Entry<String,INDArray> e : placeholders.entrySet()){
            Preconditions.checkState(sameDiff.hasVariable(e.getKey()), "Invalid placeholder passed for execution: " +
                    "No variable/placeholder with name %s exists", e.getKey());
            INDArray arr = e.getValue();
            //First: check workspaces
            if(arr.isAttached()){
                MemoryWorkspace ws = arr.data() == null ? null : arr.data().getParentWorkspace();
                if (ws != null && ws.getWorkspaceType() != MemoryWorkspace.Type.CIRCULAR) {
                    if (!ws.isScopeActive()) {
                        throw new ND4JIllegalStateException("Placeholder \"" + e.getKey() + "\" array uses leaked workspace pointer from workspace ["
                                + ws.getId() + "]: Workspace the array was defined in is no longer open.\nAll open workspaces: " + DefaultOpExecutioner.allOpenWorkspaces()
                                + "\n" + SCOPE_PANIC_MSG);
                    }

                    if (ws.getGenerationId() != arr.data().getGenerationId())
                        throw new ND4JIllegalStateException("Placeholder \"" + e.getKey() + "\" array uses outdated workspace pointer from workspace ["
                                + ws.getId() + "]: Workspace array was defined in has been closed and reopened at least once since array creation. Array WS iteration: " +
                                arr.data().getGenerationId() + ". Workspace current iteration: " +
                                ws.getGenerationId() + "\nAll open workspaces: " + DefaultOpExecutioner.allOpenWorkspaces() + "\n" + SCOPE_PANIC_MSG);
                }
            }


            //Second: cast the input to the required type
            DataType dt = sameDiff.getVariable(e.getKey()).dataType();
            if(arr.dataType() != dt){
                arr = arr.castTo(dt);
            }
            out.put(e.getKey(), arr);
        }
        return out;
    }

    @Override
    public INDArray[] getOutputs(SameDiffOp op, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                 Set<String> constAndPhInputs, List<Listener> listeners, At at, MultiDataSet batch, Set<String> allReqVariables) {
        if(listeners != null && listeners.size() > 0){
            SameDiffOp sdOp = sameDiff.getOps().get(op.getOp().getOwnName());
            for(Listener l : listeners){
                if(l.isActive(at.operation()))
                    l.preOpExecution(sameDiff, at, sdOp);
            }
        }

        INDArray[] out = getOutputsHelper(op.getOp(), outputFrameIter, opInputs, allIterInputs, constAndPhInputs);

        //Call listeners
        if(listeners != null && listeners.size() > 0){
//            SameDiffOp sdOp = sameDiff.getOps().get(op.getOp().getOwnName());

            Map<String, INDArray> namedOutsBuilder = new HashMap<>();

            for(int i = 0 ; i < out.length ; i++)
                namedOutsBuilder.put(op.outputsOfOp.get(i), out[i]);

            Map<String, INDArray> namedOuts = Collections.unmodifiableMap(namedOutsBuilder);

            for(Listener l : listeners){
                if(l.isActive(at.operation())) {
                    l.opExecution(sameDiff, at, batch, op, out);

                    for(String varName : namedOuts.keySet()){
                        l.activationAvailable(sameDiff, at, batch, op, varName, namedOuts.get(varName));
                    }
                }
            }
        }

        /*
        Deallocate input arrays if possible
        We can only do this in the following cases:
        1. The input is an ARRAY type SDVariable (not placeholder, constant, variable)
        2. The input is not one of the requested output arrays
        3. The input has been "fully consumed" - i.e., isn't required to calculate any other required ops
         */
        List<String> inputs = op.getInputsToOp();
        if(inputs != null && !inputs.isEmpty()){
            for( int i=0; i<inputs.size(); i++ ) {
                String inName = inputs.get(i);

                SDVariable v = sameDiff.getVariable(inName);
                if(v.getVariableType() != VariableType.ARRAY || allReqVariables.contains(inName)){
                    continue;
                }

                //Check if fully consumed (not required anywhere else)
                boolean canDealloc = false;
                Variable var = sameDiff.getVariables().get(inName);
                if(var.getInputsForOp().isEmpty()){
                    canDealloc = true;
                } else {
                    List<String> inputForOps = var.getInputsForOp();
                    boolean allAlreadyCalced = true;
                    for(String opName : inputForOps){
                        if(subgraphOps.contains(opName)){
                            //This op IS required. We know if it has been calculated if any of the outputs of this op are present
                            // in the set of outputs...

                            VarId vidOfInput = null;
                            for(VarId vid : opInputs){
                                if(vid.getVariable().equals(inName)){
                                    vidOfInput = vid;
                                    break;
                                }
                            }


                            String firstOutput = sameDiff.getOps().get(opName).getOutputsOfOp().get(0);
                            VarId vid = newVarId(firstOutput, vidOfInput.toFrameIter());    //TODO this doesn't work for enter op, etc!

                            allAlreadyCalced = opName.equals(op.getName()) || nodeOutputs.containsKey(vid); //Note: currently calculated op outputs haven't been added to op name

                            if(!allAlreadyCalced)
                                break;
                        }
                    }

                    canDealloc = allAlreadyCalced;
                }


                if (canDealloc) {
                    INDArray arr;
                    if (op.getOp() instanceof DynamicCustomOp) {
                        DynamicCustomOp dco = (DynamicCustomOp) op.getOp();
                        arr = dco.getInputArgument(i);
                        dco.setInputArgument(i, null);
                    } else {    //TODO
                        Op o = (Op)op.getOp();
                        if(i == 0){
                            arr = o.x();
                            o.setX(null);
                        } else {
                            arr = o.y();
                            o.setY(null);
                        }
                    }

                    log.info("Determined safe to deallocate array for: {}", inName);      //TODO FrameIter
                    mmgr.release(arr);

                    //We should also clear input references for other ops - so they aren't storing reference to closed arrays
                    //So, if we have (x -> y) and we're closing x, we should clear x from any other ops where (x -> z)
                    List<String> inputForOps = var.getInputsForOp();
                    if(inputForOps.size() > 1){
                        for(String opName : inputForOps){
                            SameDiffOp o = sameDiff.getOps().get(opName);
                            int idx = o.getInputsToOp().indexOf(inName);
                            if (op.getOp() instanceof DynamicCustomOp) {
                                DynamicCustomOp dco = (DynamicCustomOp) op.getOp();
                                dco.setInputArgument(idx, null);
                            } else {    //TODO
                                Op o2 = (Op)op.getOp();
                                if(idx == 0){
                                    o2.setX(null);
                                } else {
                                    o2.setY(null);
                                }
                            }
                        }
                    }

                    //Also clear outputs:
                    if(var.getOutputOfOp() != null){
                        SameDiffOp o = sameDiff.getOps().get(var.getOutputOfOp());

                        if( o.getOp() instanceof DynamicCustomOp){
                            DynamicCustomOp dco = (DynamicCustomOp)o.getOp();
                            List<String> outNames = o.getOutputsOfOp();
                            int idx = outNames.indexOf(var.getName());
                            dco.setOutputArgument(idx, null);
                        } else {
                            ((Op)o.getOp()).setZ(null);
                        }
                    }

                    //TODO: We should clear the input array from nodeOutputs map
                }
            }

        }


        return out;
    }

    public INDArray[] getOutputsHelper(DifferentialFunction op, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                       Set<String> constAndPhInputs){

        int totalInputs = (opInputs == null ? 0 : opInputs.size()) + (constAndPhInputs == null ? 0 : constAndPhInputs.size())
                + (allIterInputs == null ? 0 : allIterInputs.size());

        boolean constPhInput = (opInputs == null || opInputs.size() == 0) && (allIterInputs == null || allIterInputs.size() == 0);

        if(op instanceof Identity ) {
            Identity i = (Identity) op;
            String[] argNames = i.argNames();
            Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in identity op, got %s", argNames);
            VarId vid = newVarId(argNames[0], outputFrameIter);
            return new INDArray[]{nodeOutputs.get(vid)};

        } else if(op instanceof Switch) {
            Switch s = (Switch) op;
            String[] argNames = s.argNames();       //Order: input, boolean array
            VarId vidPredicate = newVarId(argNames[1], outputFrameIter);
            INDArray predicate = this.nodeOutputs.get(vidPredicate);
            Preconditions.checkState(predicate.isScalar() && predicate.dataType() == DataType.BOOL, "Expected boolean predicate: got %ndSInfo", predicate);
            VarId vid = newVarId(argNames[0], outputFrameIter);
            if (predicate.getDouble(0) == 0.0) {
                return new INDArray[]{this.nodeOutputs.get(vid), null};
            } else {
                return new INDArray[]{null, this.nodeOutputs.get(vid)};
            }
        } else if(op instanceof Enter) {
            //Enter op: forwards input to specified execution frame
            Enter e = (Enter)op;
            String[] input = e.argNames();
            Preconditions.checkState(input.length == 1, "Expected only 1 arg name for enter op: got %s", input);
            Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input for Enter op \"%s\", got %s+%s", e.getOwnName(), opInputs, constAndPhInputs);

            VarId inputVarId;
            if(constPhInput) {
                //Constant or placeholder
                inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0, null);
            } else if(allIterInputs != null && allIterInputs.size() > 0){
                inputVarId = allIterInputs.iterator().next();
            } else {
                inputVarId = opInputs.iterator().next();
            }
            INDArray enterInput = this.nodeOutputs.get(inputVarId);

            Preconditions.checkNotNull(enterInput, "Could not get enter op \"%s\" input: output variable %s - %s", e.getOwnName(), e.outputVariablesNames(), outputFrameIter);
            return new INDArray[]{enterInput};
        } else if(op instanceof Exit) {
            //Exit node forwards input to parent frame

            VarId inputVarId;
            if(constPhInput){
                //Constant or placeholder
                inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0, null);
            } else if(allIterInputs != null && allIterInputs.size() > 0){
                inputVarId = allIterInputs.iterator().next();
            } else {
                inputVarId = opInputs.iterator().next();
            }
            INDArray exitInput = this.nodeOutputs.get(inputVarId);
            return new INDArray[]{exitInput};
        } else if(op instanceof NextIteration){
            //NextIteration op: forwards its single input to the output of the current frame, but increments the iteration number
            Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input for NextIteration: got %s+%s", opInputs, constAndPhInputs);
            VarId in = (allIterInputs != null && !allIterInputs.isEmpty() ? allIterInputs.iterator().next() : opInputs.iterator().next());
            Preconditions.checkState(outputFrameIter.getFrame().equals(in.getFrame()), "Expected same frame for NextIteration input vs. output:" +
                    " got input %s, output %s", in, outputFrameIter);
            Preconditions.checkState(outputFrameIter.getIteration() == in.getIteration()+1, "Expected output iteration for NextIteration output to" +
                    " be 1 larger than the input iteration. Input: %s, output %s", in, outputFrameIter);

            INDArray inArr = this.nodeOutputs.get(in);
            return new INDArray[]{inArr};
        } else if(op instanceof If) {
            If i = (If) op;
            String[] argNames = i.argNames();       //Order should be: [boolean], true, false


            throw new UnsupportedOperationException("Execution not yet implemented for: " + op.getClass().getName());
        } else if(op instanceof Merge) {
            //Merge avairable for forward pass when any of its inputs are available. When multiple are available, behaviour
            // is undefined
            Merge m = (Merge) op;
            String[] in = sameDiff.getInputsForOp(op);
            for (String s : in) {
                VarId vid = newVarId(s, outputFrameIter);
                if (nodeOutputs.containsKey(vid)) {
                    log.trace("Returning input \"{}\" for merge node \"{}\"", m.getOwnName(), s);
                    return new INDArray[]{nodeOutputs.get(vid)};
                }
            }
            throw new IllegalStateException("Merge node " + m.getOwnName() + " has no available inputs (all inputs: " + Arrays.toString(in) +
                    ") - should not be executed at this point");
        } else if(op instanceof LoopCond) {
            //LoopCond just forwards scalar boolean to output
            LoopCond lc = (LoopCond) op;
            String[] argNames = lc.argNames();
            Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in LoopCond op, got %s", argNames);
            VarId vid = newVarId(argNames[0], outputFrameIter);
            INDArray arr = nodeOutputs.get(vid);
            Preconditions.checkNotNull(arr, "Input to LoopCond op must not be null");
            Preconditions.checkState(arr.isScalar() && arr.dataType() == DataType.BOOL, "LoopCond input must be a scalar boolean, got %ndShape");
            return new INDArray[]{arr};
        } else if(op instanceof BaseTensorOp) {
            //TensorOps - special cases...
            if (op instanceof TensorArray) {
                //Create a TensorArray
                VarId vid = newVarId(op.outputVariable().getVarName(), outputFrameIter);
                Preconditions.checkState(!tensorArrays.containsKey(vid), "TensorArray already exists for %s when executing TensorArrayV3", vid);
                tensorArrays.put(vid, new ArrayList<INDArray>());

                // Note that TensorArray has 2 outputs - a 'dummy' SDVariable that represents it, and a second output (return a scalar 0.0)
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    //TODO Proper workspace support will be added to SameDiff later
                    return new INDArray[]{Nd4j.scalar(true), Nd4j.scalar(0.0f)};
                }
            } else if (op instanceof TensorArrayRead) {
                //Do lookup and return
                //Input 0 is the TensorArray (or dummy variable that represents it). Sometimes (for import) this can be like (TensorArray -> Enter -> TensorArrayRead)
                //Input 1 is the index
                SDVariable idxSDV = op.arg(1);
                INDArray idxArr = getArray(idxSDV, opInputs, allIterInputs);
                Preconditions.checkState(idxArr.isScalar(), "TensorArrayRead input argument 1 should be scalar - has shape %ndShape", idxArr);
                int i = idxArr.getInt(0);

                SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array

                //Work out the frame/iteration:
                VarId v = (opInputs == null ? null : lookup(inTensorArray.getVarName(), opInputs, false));
                if(v == null && allIterInputs != null){
                    v = lookup(inTensorArray.getVarName(), allIterInputs, false);
                }

                Preconditions.checkState(v != null, "Could not find input %s", inTensorArray.getVarName());

                while(sameDiff.getVariableOutputOp(inTensorArray.getVarName()) instanceof Enter){
                    //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayRead
                    //TODO also TensorArrayWrite, scatter, etc??
                    inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.getVarName()).arg();
                    v = newVarId(inTensorArray.getVarName(), v.getParentFrame());
                }

                List<INDArray> list = getTensorArrays().get(v);
                Preconditions.checkState(list != null, "Could not find TensorList for %s", v);
                Preconditions.checkState(list.size() > i, "Cannot get index %s from TensorList of size %s (array not present?) - VarId=%s", i, list.size(), v);

                INDArray out = list.get(i);
                return new INDArray[]{out};
            } else if (op instanceof TensorArrayWrite) {
                //TensorArrayWrite - also has a scalar 0.0 that it returns...

                SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
                //Work out the varid (frame/iteration) of the tensor array:
                VarId tArr = (opInputs == null ? null : lookup(inTensorArray.getVarName(), opInputs, false));
                if(tArr == null && allIterInputs != null){
                    tArr = lookup(inTensorArray.getVarName(), allIterInputs, false);
                }

                Preconditions.checkState(tArr != null, "Could not find input %s", inTensorArray.getVarName());

                while(sameDiff.getVariableOutputOp(inTensorArray.getVarName()) instanceof Enter){
                    //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                    //TODO also TensorArrayScatter, etc??
                    inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.getVarName()).arg();
                    tArr = newVarId(inTensorArray.getVarName(), tArr.getParentFrame());
                }

                //Input 0 is the TensorArray (or dummy variable that represents it) - but sometimes Enter, in TensorArray -> Enter -> TensorARrayRead
                //Input 1 is the index
                //Input 2 is the value to write

                String idxName = op.arg(1).getVarName();
                SDVariable idxSDV = sameDiff.getVariable(idxName);
                INDArray idxArr = getArray(idxSDV, opInputs, allIterInputs);
                Preconditions.checkState(idxArr.isScalar(), "Index variable ID for TensorArrayWrite should be a scalar, got %ndShape", idxArr);
                int idx = idxArr.getInt(0);

                String inName = op.arg(2).getVarName();
                SDVariable inSDV = sameDiff.getVariable(inName);
                INDArray arr = getArray(inSDV, opInputs, allIterInputs);
                Preconditions.checkState(arr != null, "Could not find array for %s", inName);

                Preconditions.checkState(tensorArrays.containsKey(tArr), "Tensor array does not exist for %s", tArr);
                //TODO is this always safe to insert by index for all execution orders?
                List<INDArray> l = tensorArrays.get(tArr); //.set(idx, arr);
                while (l.size() <= idx) {
                    //Can't use set(int, E) if index >= size
                    l.add(null);
                }
                l.set(idx, arr);

                //Return dummy array
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    //TODO Proper workspace support will be added to SameDiff later
                    return new INDArray[]{Nd4j.scalar(0.0f)};
                }
            } else if (op instanceof TensorArraySize) {
                //Index 0 is the TensorArray (or dummy variable that represents it)
                SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
                //Work out the varid (frame/iteration) of the tensor array:
                VarId tArr = (opInputs == null ? null : lookup(inTensorArray.getVarName(), opInputs, false));
                if(tArr == null && allIterInputs != null){
                    tArr = lookup(inTensorArray.getVarName(), allIterInputs, false);
                }
                List<INDArray> l = tensorArrays.get(tArr);
                Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    //TODO Proper workspace support will be added to SameDiff later
                    return new INDArray[]{Nd4j.scalar(DataType.INT, l.size())};
                }
            } else if (op instanceof TensorArrayConcat) {
                SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
                VarId tArr = (opInputs == null ? null : lookup(inTensorArray.getVarName(), opInputs, false));
                if(tArr == null && allIterInputs != null){
                    tArr = lookup(inTensorArray.getVarName(), allIterInputs, false);
                }
                List<INDArray> l = tensorArrays.get(tArr);
                //TODO - empty checks. But is size 0 OK?
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    //TODO Proper workspace support will be added to SameDiff later
                    INDArray concat = Nd4j.concat(0, l.toArray(new INDArray[l.size()]));
                    return new INDArray[]{concat};
                }
            } else if (op instanceof TensorArrayGather) {
                //Input 0: the TensorArray
                //Input 1: the indices (1d integer vector)

                SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
                VarId tArr = (opInputs == null ? null : lookup(inTensorArray.getVarName(), opInputs, false));
                if(tArr == null && allIterInputs != null){
                    tArr = lookup(inTensorArray.getVarName(), allIterInputs, false);
                }
                List<INDArray> l = tensorArrays.get(tArr);
                Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

                String indicesName = op.arg(1).getVarName();
                SDVariable indicesSDV = sameDiff.getVariable(indicesName);
                INDArray idxArr = getArray(indicesSDV, opInputs, allIterInputs);
                Preconditions.checkState(idxArr.isVector(), "Indices variable for TensorArrayGather should be a vector, got %ndShape for %s", idxArr, indicesName);
                Preconditions.checkState(idxArr.dataType().isIntType(), "Indices variable for TensorArrayGather should be an integer type, got %s for array %s", idxArr.dataType(), indicesName);

                int[] idxArrInt = idxArr.toIntVector();

                //Edge case: -1 means "all"
                ArrayList<INDArray> newList = new ArrayList<>();
                if(idxArrInt.length == 1 && idxArrInt[0] == -1){
                    newList.addAll(l);
                } else {
                    for (int id : idxArrInt) {
                        Preconditions.checkState(id >=0,"Index for TensorArrayGather must be >= 0, got %s", id);
                        newList.add(l.get(id));
                    }
                }
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    //TODO Proper workspace support will be added to SameDiff later
                    INDArray out = Nd4j.pile(newList);
                    return new INDArray[]{out};
                }
            } else if (op instanceof TensorArrayScatter) {
                //Scatter values from a rank (N+1)d tensor into specific indices of the TensorArray
                //Input 0: the TensorArray
                //Input 1: the indices (1d integer vector)
                //Input 2: The values to scatter

                SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
                TensorArray ta = (TensorArray) sameDiff.getVariableOutputOp(inTensorArray.getVarName());
                VarId tArr = (opInputs == null ? null : lookup(inTensorArray.getVarName(), opInputs, false));
                if(tArr == null && allIterInputs != null){
                    tArr = lookup(inTensorArray.getVarName(), allIterInputs, false);
                }
                List<INDArray> l = tensorArrays.get(tArr);
                Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

                String indicesName = op.arg(1).getVarName();
                SDVariable indicesSDV = sameDiff.getVariable(indicesName);
                INDArray idxArr = getArray(indicesSDV, opInputs, allIterInputs);
                Preconditions.checkState(idxArr.isVector(), "Indices variable for TensorArrayScatter should be a vector, got %ndShape for %s", idxArr, indicesName);
                Preconditions.checkState(idxArr.dataType().isIntType(), "Indices variable for TensorArrayScatter should be an integer type, got %s for array %s", idxArr.dataType(), indicesName);
                int[] idxs = idxArr.toIntVector();

                String valuesName = op.arg(2).getVarName();
                SDVariable valuesSDV = sameDiff.getVariable(valuesName);
                INDArray valuesArr = getArray(valuesSDV, opInputs, allIterInputs);

                while (l.size() <= idxs.length) { //Can't use set(int, E) if index >= size
                    l.add(null);
                }

                //Edge case: idxs being [-1] means "all sub arrays" (i.e., "unstack" case)
                if(idxs.length == 1 && idxs[0] == -1){
                    idxs = ArrayUtil.range(0, (int)valuesArr.size(0));
                }

                INDArrayIndex[] idx = ArrayUtil.nTimes(valuesArr.rank(), NDArrayIndex.all(), INDArrayIndex.class);
                for (int i = 0; i < idxs.length; i++) {
                    idx[0] = NDArrayIndex.point(i);
                    INDArray get = valuesArr.get(idx).dup();
                    int outIdx = idxs[i];
                    if(valuesArr.rank() == 2 && get.rank() == 2){
                        //Workaround for: https://github.com/deeplearning4j/deeplearning4j/issues/7092
                        get = get.reshape(get.length());
                    }
                    if(valuesArr.rank() == 1 && get.rank() > 0){
                        get = get.reshape(new long[0]);
                    }
                    l.set(outIdx, get);
                }

                //Return dummy array
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    //TODO Proper workspace support will be added to SameDiff later
                    return new INDArray[]{Nd4j.scalar(0.0f)};
                }
            } else if (op instanceof TensorArraySplit) {
                //Split values from a rank (N+1)d tensor into sequential indices of the TensorArray
                //For example, orig=[8,2] sizearray with split (4,4) means TensorArray[0] = orig[0:4,:] and TensorArray[1] = orig[4:8,:]
                //Input 0: the TensorArray
                //Input 1: The values to split
                //Input 2: the size of each split (1d integer vector)

                SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
                VarId tArr = (opInputs == null ? null : lookup(inTensorArray.getVarName(), opInputs, false));
                if(tArr == null && allIterInputs != null){
                    tArr = lookup(inTensorArray.getVarName(), allIterInputs, false);
                }
                List<INDArray> l = tensorArrays.get(tArr);
                Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

                String splitName = op.arg(1).getVarName();
                INDArray splitArr = getArray(sameDiff.getVariable(splitName), opInputs, allIterInputs);


                String sizeName = op.arg(2).getVarName();
                SDVariable sizeSDV = sameDiff.getVariable(sizeName);
                INDArray sizeArr = getArray(sizeSDV, opInputs, allIterInputs);
                Preconditions.checkState(sizeArr.isVector(), "Indices variable for TensorArraySplit should be a vector, got %ndShape for %s", sizeArr, sizeName);
                Preconditions.checkState(sizeArr.dataType().isIntType(), "Indices variable for TensorArraySplit should be an integer type, got %s for array %s", sizeArr.dataType(), sizeName);
                int[] sizes = sizeArr.toIntVector();

                while (l.size() <= sizes.length) { //Can't use set(int, E) if index >= size
                    l.add(null);
                }

                INDArrayIndex[] idx = ArrayUtil.nTimes(splitArr.rank(), NDArrayIndex.all(), INDArrayIndex.class);
                int soFar = 0;
                for (int i = 0; i < sizes.length; i++) {
                    idx[0] = NDArrayIndex.interval(soFar, soFar + sizes[i]);
                    INDArray sub = splitArr.get(idx).dup();
                    l.set(i, sub);
                    soFar += sizes[i];
                }
                //Return dummy array
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    //TODO Proper workspace support will be added to SameDiff later
                    return new INDArray[]{Nd4j.scalar(0.0f)};
                }
            } else {
                throw new IllegalStateException("Execution support not yet implemented for: " + op.getClass().getName());
            }
        } else if(op instanceof GradientBackwardsMarker){
            return new INDArray[]{Nd4j.scalar(1.0f)};
        } else if(op instanceof CustomOp){
            CustomOp c = (CustomOp)op;
            Nd4j.getExecutioner().exec(c);
            return c.outputArguments();
        } else if(op instanceof Op) {
            Op o = (Op) op;
            Nd4j.getExecutioner().exec(o);
            return new INDArray[]{o.z()};
        } else {
            throw new UnsupportedOperationException("Execution not yet implemented for: " + op.getClass().getName());
        }
    }

    @Override
    public INDArray getConstantOrVariable(String variableName) {
        SDVariable v = sameDiff.getVariable(variableName);
        Preconditions.checkState(sameDiff.getVariable(variableName).isConstant() || v.getVariableType() == VariableType.VARIABLE,
                "Variable %s is not a constant", variableName);
        return sameDiff.getArrForVarName(variableName);
    }

    @Override
    public SameDiffOp getAndParameterizeOp(String opName, FrameIter frameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                                     Set<String> constAndPhInputs, Map<String,INDArray> placeholderValues, Set<String> allReqVariables) {

//        DifferentialFunction df = sameDiff.getOpById(opName);
        SameDiffOp sdo = sameDiff.getOps().get(opName);
        DifferentialFunction df = sdo.getOp();

        //TODO We should clone these ops - probably - as we don't want them shared between threads/sessions!
        //But let's only clone them *once* and cache in inference session - not on every exec

        Preconditions.checkNotNull(df, "No differential function fond with name %s", opName);

        if(df instanceof LoopCond || df instanceof Enter || df instanceof Exit || df instanceof NextIteration ||
                df instanceof Merge || df instanceof Switch || df instanceof If || df instanceof While ||
                df instanceof BaseTensorOp){
            //Control dependencies and tensor ops (like TensorArray, TensorArrayRead etc) don't need inputs set, execution is a special case
            return sdo;
        }

        //Infer the args based on the inputs (variable + frame + iteration)
        String[] argNames = df.argNames();
        int numArgs = (argNames == null ? 0 : argNames.length);
        int numNonConstIns = (opInputs == null ? 0 : opInputs.size());
        int numNonConstInsAllIters = (allIterInputs == null ? 0 : allIterInputs.size());
        int numConstPhIns = (constAndPhInputs == null ? 0 : constAndPhInputs.size());

        Set<String> constEnterInputs = null;
        if(numArgs != (numNonConstIns + numConstPhIns + numNonConstInsAllIters)){
            boolean anyConstEnterInputs = false;
            SDVariable[] args = df.args();
            for(SDVariable v : args){
                Variable var = sameDiff.getVariables().get(v.getVarName());
                //Nested enter case:
                DifferentialFunction inputVarFn = (var.getOutputOfOp() == null ? null : sameDiff.getOps().get(var.getOutputOfOp()).getOp());
                if(inputVarFn instanceof Enter && ((Enter)inputVarFn).isConstant()){
                    anyConstEnterInputs = true;
                    if(constEnterInputs == null)
                        constEnterInputs = new HashSet<>();
                    constEnterInputs.add(v.getVarName());
                }
            }

            int constEnterInputCount = 0;
            if(anyConstEnterInputs){
                /*
                2019/01/26: AB
                Resolve nested enter inputs (constants 2+ enters in)
                Why this hack is necessary: consider the following (sub) graph:     constX -> Enter(a) -> Enter(b) -> opY
                On iterations (a=0, b=0) all is well, opY gets triggered as normal.
                On iterations (a>0, b=*) the "opY is available for exec" won't be triggered.
                This is because Enter(a) is only executed once, on iteration 0 of the outer loop.
                Consequently, Enter(b) is not triggered as available on iteration 1+.
                When we do the lookup for the actual array to use for op execution (i.e., get inputs for opY(a=1,b=0))
                it won't be found.
                This is a bit of an ugly hack, though I've yet to find a cleaner solution.
                It should only be required with the combination of: constants, 2 levels of enters, and more than 1 iteration in each loop.
                 */

                //For example, const -> Enter(a) -> Enter(b) -> op; in this case, the input to Op (at any frame/iteration) should should
                // be the constant value - which is recorded as (frame="a",iter=0,parent=(frame="b",iter=0))
                for(String s : constEnterInputs){
                    //First: check if this has already been provided
                    if(constAndPhInputs != null && constAndPhInputs.contains(s)){
                        //already resolved/provided
                        continue;
                    }
                    boolean found = false;
                    if(allIterInputs != null) {
                        for (VarId vid : allIterInputs) {
                            if (s.equals(vid.getVariable())) {
                                //Already resolved/provided
                                found = true;
                                break;
                            }
                        }
                    }
                    if(found)
                        continue;

                    constEnterInputCount++;
                }
            }

            if(numArgs > 1){
                //Might be due to repeated inputs
                Set<String> uniqueArgNames = new HashSet<>();
                Collections.addAll(uniqueArgNames, argNames);
                Preconditions.checkState(uniqueArgNames.size() == (numNonConstIns + numConstPhIns + numNonConstInsAllIters + constEnterInputCount),
                        "Different number of arg names as op inputs for op %s (%s): arg names %s vs. op inputs %s+%s", df.getClass().getSimpleName(),
                        opName, uniqueArgNames, opInputs, constAndPhInputs);
            } else {
                Preconditions.checkState(numArgs == (numNonConstIns + numConstPhIns + constEnterInputCount),
                        "Different number of arg names as op inputs for op %s (%s): arg names %s vs. op inputs %s+%s", df.getClass().getSimpleName(),
                        opName, argNames, opInputs, constAndPhInputs);
            }
        }

        INDArray[] args = null;
        if(argNames != null && argNames.length > 0) {
            args = new INDArray[argNames.length];
            int i = 0;
            for(String s : argNames){
                SDVariable v = sameDiff.getVariable(s);
                if(v.isConstant()) {
                    args[i] = v.getArr();
                } else if(v.isPlaceHolder()) {
                    Preconditions.checkState(placeholderValues != null && placeholderValues.containsKey(s), "No array provided for placeholder %s", s);
                    args[i] = placeholderValues.get(s);
                } else if(constEnterInputs != null && constEnterInputs.contains(s)){
                    //For enter nodes that are constants, we want iteration 0 in all frames in the heirarchy
                    //For example, const -> Enter(a) -> Enter(b) -> op; in this case, the input to Op (at any frame/iteration) should should
                    // be the constant value - which is recorded as (frame="a",iter=0,parent=(frame="b",iter=0))
                    VarId vid = newVarId(s, frameIter.clone());
                    vid.setIteration(0);
                    FrameIter toZero = vid.getParentFrame();
                    while(toZero != null){
                        toZero.setIteration(0);
                        toZero = toZero.getParentFrame();
                    }
                    INDArray arr = this.nodeOutputs.get(vid);
                    args[i] = arr;
                } else {
                    if(opInputs != null) {
                        for (VarId vid : opInputs) {
                            if (vid.getVariable().equals(s)) {
                                args[i] = this.nodeOutputs.get(vid);
                                break;
                            }
                        }
                    }
                    if(args[i] == null && allIterInputs != null){
                        for(VarId vid : allIterInputs){
                            if(vid.getVariable().equals(s)){
                                args[i] = this.nodeOutputs.get(vid);
                                break;
                            }
                        }
                    }
                }
                Preconditions.checkNotNull(args[i], "Could not parameterize op %s: array %s (variable %s) is null", opName, i, v.getVarName());
                i++;
            }

        }

        //Set the op inputs and output arguments
        //Note that when we are in a loop (and non-first iteration), we want to allocate new arrays even if shapes are
        // ok: this is because we need the values in past iterations for backprop (potentially)
        //TODO let's find a way to use in-place modification for loops where possible to reduce memory requirements
        boolean isLoop = !frameIter.getFrame().equals(OUTER_FRAME) && frameIter.getIteration() > 0;

        if(df instanceof CustomOp){
            DynamicCustomOp customOp = (DynamicCustomOp) df;
            if(args != null) {
                customOp.setInputArguments(args);
            }

            df.resolvePropertiesFromSameDiffBeforeExecution();
            List<LongShapeDescriptor> outShape = customOp.calculateOutputShape();
            Preconditions.checkState(outShape != null && outShape.size() > 0, "Failed to calculate output shapes for op %s (%s) - no shapes were returned by calculateOutputShape()", customOp.opName(), customOp.getOwnName());
            String[] outNames = df.outputVariablesNames();
            Preconditions.checkState(outNames.length == outShape.size(), "Error in operation shape calculation for op \"%s\": Got %s op output shapes for an operation" +
                    " with %s outputs (number of shapes and outputs must be equal)", df.opName(), outShape.size(), outNames.length);
            for( int i=0; i<outShape.size(); i++ ){
                INDArray currOutput = (customOp.numOutputArguments() <= i ? null : customOp.getOutputArgument(i));
                LongShapeDescriptor reqShape = outShape.get(i);

                //Issue: many ops have multiple valid output datatypes, and output shape calc can't at present know which: https://github.com/deeplearning4j/deeplearning4j/issues/6872
                //As a workaround, we'll use the output variable datatype instead.
                DataType dt = sameDiff.getVariable(outNames[i]).dataType();
                DataType currDT = reqShape.dataType();
                if(dt != currDT){
                    reqShape = reqShape.asDataType(dt);
                }

                if(currOutput == null || !currOutput.shapeDescriptor().equals(reqShape) || currOutput.isEmpty() != reqShape.isEmpty() || isLoop){
//                    INDArray out;
//                    try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
//                        //TODO Proper workspace support will be added to SameDiff later
//                        out = Nd4j.create(reqShape, false);
//                    }

                    boolean isOutput = allReqVariables.contains(outNames[i]);
                    INDArray out = mmgr.allocate(isOutput, reqShape);
                    customOp.setOutputArgument(i, out);
                }
            }

        } else if(df instanceof Op){
            Op op = (Op) df;

            boolean axisArg = false;
            boolean emptyReduce = false;
            if(op instanceof ReduceOp && ((ReduceOp) op).getOpType() != Op.Type.REDUCE3 && df.argNames().length == 2){
                //2nd input should be treated as integer axis arg...
                SDVariable axisArgVar = df.arg(1);
                Preconditions.checkState(axisArgVar.dataType().isIntType(), "Legacy op %s input 1 (axis) was expected to be an integer type, is %s", df.getClass(), axisArgVar.dataType());

                INDArray arr = getArray(axisArgVar, opInputs, allIterInputs);
                Preconditions.checkState(arr != null, "Could not get axis argument for op %s: %s", df.getOwnName(), df.getClass());
                if(!arr.isEmpty()){
                    int[] axis = arr.toIntVector();
                    int rank = args[0].rank();
                    axis = Shape.normalizeAxis(rank, axis);
                    df.setDimensions(axis);
                    ((BaseReduceOp)op).setEmptyReduce(false);
                } else {
                    df.setDimensions(null);
                    emptyReduce = true;
                    //Note: edge case: [x,y].sum(empty) = [x,y] for TF import compatibility.
                    //Note also that empty is not the same as int[0] as in INDArray.sum(new int[0])
                    ((BaseReduceOp)op).setEmptyReduce(true);
                }
                axisArg = true;
            } else if(op instanceof ScalarOp && df.argNames().length == 2){
                //Scalar ops: 2nd input should be treated as scalar...
                SDVariable scalarVar = df.arg(1);
                INDArray scalar = getArray(scalarVar, opInputs, allIterInputs);
                Preconditions.checkState(scalar != null, "Could not get scalar argument for op %s: %s", df.getOwnName(), df.getClass());
                Preconditions.checkState(scalar.isScalar(), "Scalar argument for op %s (%s) is not a scalar: has shape %ndShape", df.getOwnName(), df.getClass(), scalar );
                ((ScalarOp) op).setScalar(scalar);
            }

            if(args != null && args.length > 0){
                op.setX(args[0]);
                if (args.length == 2 && !axisArg)
                    op.setY(args[1]);
            }


            //Check output shape; allocate a new Z if required
            //For example, if minibatch size has changed since last op execution
            if(emptyReduce){
                INDArray z = op.z();
                if (z == null || !op.x().equalShapes(z) || isLoop) {
                    //Note: edge case: [x,y].sum(empty) = [x,y] for TF import compatibility.
                    op.setZ(op.x().ulike());
                }
            } else {
                List<LongShapeDescriptor> outputShape = ((BaseOp) op).calculateOutputShape();
                Preconditions.checkState(outputShape != null && outputShape.size() == 1, "Could not calculate output shape for op: %s", op.getClass());
                INDArray z = op.z();
                if (z == null || !outputShape.get(0).equals(z.shapeDescriptor()) || isLoop) {
                    if (log.isTraceEnabled()) {
                        log.trace("Existing op result (z) array shape for op {} was {}, allocating new array of shape {}",
                                op.getClass().getSimpleName(), (z == null ? null : Arrays.toString(z.shape())), outputShape.get(0).toString());
                    }

                    LongShapeDescriptor lsd = outputShape.get(0);
//                    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
//                        //TODO Proper workspace support will be added to SameDiff later
//                        z = Nd4j.create(lsd, false);
//                    }

                    boolean isOutput = allReqVariables.contains(((BaseOp) op).outputVariablesNames()[0]);
                    z = mmgr.allocate(isOutput, lsd);
                    op.setZ(z);
                }
            }
            df.resolvePropertiesFromSameDiffBeforeExecution();
        }

        return sdo;
    }


    protected INDArray getArray(SDVariable sdv, Collection<VarId> opInputs, Collection<VarId> allIterInputs){
        String n = sdv.getVarName();
        if(sdv.getVariableType() == VariableType.CONSTANT || sdv.getVariableType() == VariableType.VARIABLE){
            return getConstantOrVariable(n);
        } else {
            VarId inVarId = null;
            if(opInputs != null){
                inVarId = lookup(n, opInputs, false);
            }
            if(inVarId == null && allIterInputs != null && !allIterInputs.isEmpty()){
                inVarId = lookup(n, allIterInputs, false);
            }
            Preconditions.checkState(inVarId != null,"Could not find array for variable %s", sdv.getVarName());
            return nodeOutputs.get(inVarId);
        }
    }
}
