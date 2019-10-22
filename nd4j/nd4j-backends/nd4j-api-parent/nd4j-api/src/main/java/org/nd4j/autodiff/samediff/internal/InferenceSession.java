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

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCloseMemoryMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.Stack;
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
 * InferenceSession: Performs inference (forward pass) on a SameDiff instance to get the outputs of the requested nodes.<br>
 * Dynamically (in AbstractSession) calculates the required subgraph to execute to get the required outputs.<br>
 * Note that while AbstractSession handles the graph structure component, InferenceSession handles only op execution
 * and memory management<br>
 * <br>
 * For INDArray memory management - i.e., tracking and releasing memory manually, as soon as possible, to
 * minimize memory use - this is implemented using a {@link SessionMemMgr} instance (for allocations/deallocations) and
 * also {@link IdentityDependencyTracker} to track where arrays are actually used. The IdentityDependencyTracker tells
 * us when the array is no longer needed (i.e., has been "fully consumed" by ops depending on it) accounting for the
 * fact that some operations, such as identity, enter, exit, etc, are "zero copy" for performance reasons.
 *
 * @author Alex Black
 */
@Slf4j
public class InferenceSession extends AbstractSession<INDArray,SameDiffOp> {
    private static final String SCOPE_PANIC_MSG = "If required, arrays in workspaces can be detached using INDArray.detach() before being passed to the SameDiff instance.\n" +
            "Alternatively, arrays defined in a workspace must be replaced after the workspace has been closed.";

    protected static final String KERAS_TRAIN_TEST = "keras_learning_phase";

    @Getter @Setter
    private SessionMemMgr mmgr;     //Used for allocating and deallocating memory
    /**
     * Array use tracker: What needs to happen before the array can be closed/released?
     * As the name suggests, the INDArrays are tracked using qbject identity, not equality
     */
    @Getter @Setter
    private IdentityDependencyTracker<INDArray, Dep> arrayUseTracker = new IdentityDependencyTracker<>();


    public InferenceSession(@NonNull SameDiff sameDiff) {
        super(sameDiff);

        mmgr = new ArrayCloseMemoryMgr();   //TODO replace this with new (planned) array reuse memory manager
    }

    @Override
    protected Map<String,INDArray> preprocessPlaceholders(Map<String,INDArray> placeholders, At at){
        arrayUseTracker.clear();

        //We'll also use this method as a "pre execution" hook-in, to mark variables as something we should never deallocate
        //This occurs by never marking these "ConstantDep" and "VariableDep" instances as satisfied, so there's always
        // an unsatisfied dependency for them in the array use tracker
        //TODO we shouldn't be clearing this on every single iteration, in 99.5% of cases variables will be same as last iteration...
        for(SDVariable v : sameDiff.variables()){
            if(v.getVariableType() == VariableType.CONSTANT){
                arrayUseTracker.addDependency(v.getArr(), new ConstantDep(v.getVarName()));
            } else if(v.getVariableType() == VariableType.VARIABLE){
                arrayUseTracker.addDependency(v.getArr(), new VariableDep(v.getVarName()));
            }
        }

        //Workaround for some TF/Keras based models that require explicit train/test as a placeholder
        boolean kerasWorkaround = false;
        List<String> phs = sameDiff.inputs();
        if(phs != null && !phs.isEmpty()) {
            for(String s : phs) {
                if (s.endsWith(KERAS_TRAIN_TEST) && !placeholders.containsKey(s)) {
                    // The behaviour of some Keras layers (like GRU) differs depending on whether the model is training.
                    // We provide this value directly, unless the user has provided this manually
                    INDArray scalar = mmgr.allocate(false, DataType.BOOL).assign(at.operation().isTrainingPhase());
                    placeholders = new HashMap<>(placeholders); //Array might be singleton, or otherwise unmodifiable
                    placeholders.put(s, scalar);
                    kerasWorkaround = true;
                }
            }
        }


        if(placeholders == null || placeholders.isEmpty()){
            return placeholders;
        }

        //Handle casting of the input array automatically.
        //The idea here is to avoid unexpected errors if the user (for example) tries to perform inference with a double
        // array for a float placeholder
        //TODO eventually we might have ops that support multiple input types, and hence won't need this casting
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
            //TODO For the casting case, we SHOULD actually deallocate this when we're done with it, which is usually sooner than "exec done"
            DataType dt = sameDiff.getVariable(e.getKey()).dataType();
            if(kerasWorkaround && e.getKey().endsWith(KERAS_TRAIN_TEST)) {
                arrayUseTracker.addDependency(arr, new ExecDoneDep());
            } else if(arr.dataType() == dt){
                //Mark as a placeholder array in the array use tracker, so we never deallocate this array...
                arrayUseTracker.addDependency(e.getValue(), new PlaceholderDep(e.getKey()));
            } else {
                INDArray cast = mmgr.allocate(false, dt, arr.shape());
                cast.assign(arr);
                arr = cast;
                //This array CAN be deallocated once consumed, because of the cast
                //TODO we can likely close this sooner
                arrayUseTracker.addDependency(arr, new ExecDoneDep());
            }
            out.put(e.getKey(), arr);
        }

        return out;
    }

    @Override
    protected Map<String,INDArray> postProcessOutput(Map<String,INDArray> output){

        //For any queued (not yet processed) ops - mark them as satisfied, so we can deallocate any arrays
        // that are waiting on them
        if(dt.hasNewAllSatisfied()){
            List<ExecStep> execSteps = dt.getNewAllSatisfiedList();
            for(ExecStep es : execSteps){
                if(es.getType() == ExecType.OP){
                    OpDep od = new OpDep(es.getName(), es.getFrameIter().getFrame(), es.getFrameIter().getIteration(), es.getFrameIter().getParentFrame());
                    arrayUseTracker.markSatisfied(od, true);
                }
            }
        }

        //Also mark "end of execution" for array dependency tracker. Mainly used for TensorArray arrays at present.
        //TODO Optimize for reduced memory for some TensorArray operations - i.e., close/deallocate earlier
        arrayUseTracker.markSatisfied(new ExecDoneDep(), true);
        if(arrayUseTracker.hasNewAllSatisfied()){
            List<INDArray> l = arrayUseTracker.getNewAllSatisfiedList();
            for(INDArray arr : l){
                mmgr.release(arr);
            }
        }

        return output;
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

        INDArray[] out = doExec(op.getOp(), outputFrameIter, opInputs, allIterInputs, constAndPhInputs);
        op.getOp().clearArrays();

        StringBuilder sb = new StringBuilder();
        sb.append(op.getName()).append(" - ").append(outputFrameIter).append(" outputs: ");
        List<String> opOutNames = op.getOutputsOfOp();
        for( int i=0; i<out.length; i++ ){
            if(i > 0)
                sb.append(", ");
            sb.append("(").append(i).append(" - ").append(opOutNames.get(i)).append(" = ").append(
                    out[i] == null ? null : out[i].getId()).append(")");
        }
        System.out.println(sb.toString());

        //Call listeners, before we (maybe) deallocate input arrays
        if(listeners != null && listeners.size() > 0){
            Map<String, INDArray> namedOuts = null;

            for(Listener l : listeners){
                if(l.isActive(at.operation())) {
                    //Lazily create map, only if required
                    if(namedOuts == null){
                        Map<String, INDArray> namedOutsBuilder = new HashMap<>();

                        for(int i = 0 ; i < out.length ; i++)
                            namedOutsBuilder.put(op.outputsOfOp.get(i), out[i]);
                        namedOuts = Collections.unmodifiableMap(namedOutsBuilder);
                    }


                    l.opExecution(sameDiff, at, batch, op, out);

                    for(String varName : namedOuts.keySet()){
                        l.activationAvailable(sameDiff, at, batch, op, varName, namedOuts.get(varName));
                    }
                }
            }
        }


        //Record array uses for memory management/deallocation
        SameDiffOp o = sameDiff.getOps().get(op.getName());
        List<String> outVarNames = o.getOutputsOfOp();
        for( int i=0; i<out.length; i++ ){
            if(out[i] == null && o.getOp() instanceof Switch)
                continue;   //Switch case: we only ever get one of 2 outputs, other is null (branch not executed)

            String name = outVarNames.get(i);
            Variable v = sameDiff.getVariables().get(name);
            List<String> inputsForOps = v.getInputsForOp();
            if(inputsForOps != null){
                for(String opName : inputsForOps){
                    //Only add dependencies if we actually need the op this feeds into, otherwise the dependency
                    // will will never be marked as satisfied
                    if(!subgraphOps.contains(opName))
                        continue;

                    SameDiffOp forOp = sameDiff.getOps().get(opName);

                    //TODO do switch or merge need special handling also?
                    if(forOp.getOp() instanceof Enter) {
                        Enter e = (Enter)forOp.getOp();
                        if(e.isConstant()) {
                        /*
                        Contant enter case: Need to keep this array around for the entire duration of the frame, including
                        any nested frames, and all iterations.
                        Unfortunately, we don't know exactly when we're done with a frame for good
                        This isn't a great solution, but other possibilities (frame close, trying to detect all exit ops,
                        detecting return to parent frame, etc all fail in certain circumstances, such as due to control dependencies
                        on variables).
                         */
                            Dep d = new ExecDoneDep();
                            arrayUseTracker.addDependency(out[i], d);
                        } else {
                            Dep d = new OpDep(opName, e.getFrameName(), 0, outputFrameIter);
                            arrayUseTracker.addDependency(out[i], d);       //Op defined by "d" needs to be executed before specified array can be closed
                        }
                    } else if(forOp.getOp() instanceof NextIteration) {
                        //The array is needed by the NEXT iteration op, not the current one
                        Dep d = new OpDep(opName, outputFrameIter.getFrame(), outputFrameIter.getIteration() + 1, outputFrameIter.getParentFrame());
                        arrayUseTracker.addDependency(out[i], d);
                    } else if(forOp.getOp() instanceof Exit){
                        //The array is needed at the EXIT frame (i.e., parent frame), not the inner/just executed one
                        FrameIter fi = outputFrameIter.getParentFrame();
                        Dep d = new OpDep(opName, fi.getFrame(), fi.getIteration(), fi.getParentFrame());
                        arrayUseTracker.addDependency(out[i], d);       //Op defined by "d" needs to be executed before specified array can be closed
                    } else {
                        //All other ops...
                        Dep d = new OpDep(opName, outputFrameIter.getFrame(), outputFrameIter.getIteration(), outputFrameIter.getParentFrame());
                        arrayUseTracker.addDependency(out[i], d);       //Op defined by "d" needs to be executed before specified array can be closed
                    }
                }
            }

            if(OUTER_FRAME.equals(outputFrameIter.getFrame()) && allReqVariables.contains(name)){
                //This variable is an output, record that in the array use tracker, so we don't deallocate it
                arrayUseTracker.addDependency(out[i], new ReqOutputDep(name));
            } else if((inputsForOps == null || inputsForOps.isEmpty()) && !arrayUseTracker.hasDependency(out[i])){
                //This particular array is not actually needed anywhere, so we can deallocate in immediately
                //Possibly only a control dependency, or only one of the outputs of a multi-output op is used
                log.info("Found array id {} (output of {}) not required anywhere, deallocating", out[i].getId(), o.getName());
                mmgr.release(out[i]);
            }
        }

        //Mark current op dependency as satisfied...
        Dep d = new OpDep(op.getName(), outputFrameIter.getFrame(), outputFrameIter.getIteration(), outputFrameIter.getParentFrame());
        arrayUseTracker.markSatisfied(d, true);


        //Close any no longer required arrays
        if(arrayUseTracker.hasNewAllSatisfied()){
            List<INDArray> canClose = arrayUseTracker.getNewAllSatisfiedList();
            for(INDArray arr : canClose){
                log.info("Closing array... id={}, {}", arr.getId(), arr.shapeInfoToString());
                mmgr.release(arr);
            }
        }

        return out;
    }

    public INDArray[] doExec(DifferentialFunction op, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                             Set<String> constAndPhInputs) {

        int totalInputs = (opInputs == null ? 0 : opInputs.size()) + (constAndPhInputs == null ? 0 : constAndPhInputs.size())
                + (allIterInputs == null ? 0 : allIterInputs.size());

        boolean constPhInput = (opInputs == null || opInputs.size() == 0) && (allIterInputs == null || allIterInputs.size() == 0);

        if(op instanceof Identity ) {
            Identity i = (Identity) op;
            String[] argNames = i.argNames();
            Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in identity op, got %s", argNames);
            VarId vid = outputFrameIter.toVarId(argNames[0]);

            INDArray orig = nodeOutputs.get(vid);
            return new INDArray[]{orig};
        } else if(op instanceof Switch) {
            Switch s = (Switch) op;
            String[] argNames = s.argNames();       //Order: input, boolean array
            VarId vidPredicate = outputFrameIter.toVarId(argNames[1]);
            INDArray predicate = this.nodeOutputs.get(vidPredicate);
            Preconditions.checkState(predicate.isScalar() && predicate.dataType() == DataType.BOOL, "Expected boolean predicate: got %ndSInfo", predicate);
            VarId vid = outputFrameIter.toVarId(argNames[0]);
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
            if(inArr == null) {
                Preconditions.throwStateEx("Could not find array for NextIteration operation %s with output %s (frame=%s, iteration=%s)",
                        op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
            }
            return new INDArray[]{inArr};
        } else if(op instanceof Merge) {
            //Merge available for forward pass when any of its inputs are available. When multiple are available, behaviour
            // is undefined
            Merge m = (Merge) op;
            String[] in = sameDiff.getInputsForOp(op);
            for (String s : in) {
                VarId vid = outputFrameIter.toVarId(s);
                if (nodeOutputs.containsKey(vid)) {
                    log.trace("Returning input \"{}\" for merge node \"{}\"", m.getOwnName(), s);
                    INDArray arr = nodeOutputs.get(vid);
                    Preconditions.checkState(arr != null, "Could not find output array for %s", vid);
                    return new INDArray[]{arr};
                }
            }
            throw new IllegalStateException("Merge node " + m.getOwnName() + " has no available inputs (all inputs: " + Arrays.toString(in) +
                    ") - should not be executed at this point");
        } else if(op instanceof LoopCond) {
            //LoopCond just forwards scalar boolean to output
            LoopCond lc = (LoopCond) op;
            String[] argNames = lc.argNames();
            Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in LoopCond op, got %s", argNames);
            VarId vid = outputFrameIter.toVarId(argNames[0]);
            INDArray arr = nodeOutputs.get(vid);
            Preconditions.checkNotNull(arr, "Input to LoopCond op must not be null");
            Preconditions.checkState(arr.isScalar() && arr.dataType() == DataType.BOOL, "LoopCond input must be a scalar boolean, got %ndShape");
            return new INDArray[]{arr};
        } else if(op instanceof BaseTensorOp) {
            //TensorOps - special cases...
            return getOutputsHelperTensorArrayOps(op, outputFrameIter, opInputs, allIterInputs);
        } else if(op instanceof GradientBackwardsMarker) {
            INDArray out = mmgr.allocate(false, DataType.FLOAT).assign(1.0f);
            return new INDArray[]{out};
        } else if(op instanceof ExternalErrorsFunction){
            ExternalErrorsFunction fn = (ExternalErrorsFunction)op;
            String n = fn.getGradPlaceholderName();
            INDArray arr = nodeOutputs.get(new VarId(n, OUTER_FRAME, 0, null));
            Preconditions.checkState(arr != null, "Could not find external errors placeholder array: %s", arr);
            INDArray out = mmgr.allocate(false, arr.dataType(), arr.shape());
            out.assign(arr);
            return new INDArray[]{out};
        } else if(op instanceof CustomOp){
            CustomOp c = (CustomOp)op;
            Nd4j.exec(c);
            return c.outputArguments();
        } else if(op instanceof Op) {
            Op o = (Op) op;
            Nd4j.exec(o);
            return new INDArray[]{o.z()};
        } else {
            throw new UnsupportedOperationException("Execution not yet implemented for: " + op.getClass().getName());
        }
    }

    /**
     * Forward pass for TensorArray ops
     */
    public INDArray[] getOutputsHelperTensorArrayOps(DifferentialFunction op, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs){
        /*
        TODO: TensorArray memory management note: For now, we'll close any INDArrays stored in the TensorArray at the end of
        graph execution. This uses more memory than necessary for an earlier close strategy, but simplifies memory management.
        This should be revisited and optimized later
         */

        if (op instanceof TensorArray) {
            //Create a TensorArray
            VarId vid = outputFrameIter.toVarId(op.outputVariable().getVarName());
            Preconditions.checkState(!tensorArrays.containsKey(vid), "TensorArray already exists for %s when executing TensorArrayV3", vid);
            tensorArrays.put(vid, new ArrayList<INDArray>());

            // Note that TensorArray has 2 outputs - a 'dummy' SDVariable that represents it, and a second output (return a scalar 0.0)
            INDArray dummy = mmgr.allocate(false, DataType.BOOL).assign(true);
            INDArray scalar = mmgr.allocate(false, DataType.FLOAT).assign(0.0);
            return new INDArray[]{dummy, scalar};
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
                v = v.getParentFrame().toVarId(inTensorArray.getVarName());
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
                tArr = tArr.getParentFrame().toVarId(inTensorArray.getVarName());
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

            //Add a dependency
            Dep d = new ExecDoneDep();
            arrayUseTracker.addDependency(arr, d);

            //Return dummy array
            INDArray scalar = mmgr.allocate(false, DataType.FLOAT).assign(0.0);
            return new INDArray[]{scalar};
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

            INDArray scalar = mmgr.allocate(false, DataType.INT).assign(l.size());
            return new INDArray[]{scalar};
        } else if (op instanceof TensorArrayConcat) {
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.getVarName(), opInputs, false));
            if(tArr == null && allIterInputs != null){
                tArr = lookup(inTensorArray.getVarName(), allIterInputs, false);
            }
            List<INDArray> l = tensorArrays.get(tArr);

            Concat c = new Concat(0, l.toArray(new INDArray[0]));
            List<LongShapeDescriptor> shape = c.calculateOutputShape();
            INDArray out = mmgr.allocate(false, shape.get(0));
            c.setOutputArgument(0, out);
            Nd4j.exec(c);
            return new INDArray[]{out};
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
            List<INDArray> newList = new ArrayList<>();
            if(idxArrInt.length == 1 && idxArrInt[0] == -1){
                newList.addAll(l);
            } else {
                for (int id : idxArrInt) {
                    Preconditions.checkState(id >=0,"Index for TensorArrayGather must be >= 0, got %s", id);
                    newList.add(l.get(id));
                }
            }

            Stack s = new Stack(newList.toArray(new INDArray[0]), null, 0);
            List<LongShapeDescriptor> shape = s.calculateOutputShape();
            INDArray out = mmgr.allocate(false, shape.get(0));
            s.setOutputArgument(0, out);
            Nd4j.exec(s);
            return new INDArray[]{out};
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
                INDArray get = mmgr.dup(valuesArr.get(idx));
                int outIdx = idxs[i];
                if(valuesArr.rank() == 1 && get.rank() > 0){
                    get = get.reshape();
                }
                l.set(outIdx, get);

                //Add dependency for values array until end of execution
                arrayUseTracker.addDependency(get, new ExecDoneDep());
            }

            //Return dummy array
            INDArray scalar = mmgr.allocate(false, DataType.FLOAT).assign(0.0);
            return new INDArray[]{scalar};
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
                INDArray sub = mmgr.dup(splitArr.get(idx));
                l.set(i, sub);
                soFar += sizes[i];

                //Add dependency for values array until end of execution
                arrayUseTracker.addDependency(sub, new ExecDoneDep());
            }

            //Return dummy array
            INDArray scalar = mmgr.allocate(false, DataType.FLOAT).assign(0.0);
            return new INDArray[]{scalar};
        } else {
            throw new IllegalStateException("Execution support not yet implemented for: " + op.getClass().getName());
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
        SameDiffOp sdo = sameDiff.getOps().get(opName);
        DifferentialFunction df = sdo.getOp();

        //TODO Switch to OpContext - and make sure executing like that is thread safe (i.e., array fields in ops are not used etc)

        Preconditions.checkNotNull(df, "No differential function found with name \"%s\"", opName);

        if(df instanceof LoopCond || df instanceof Enter || df instanceof Exit || df instanceof NextIteration ||
                df instanceof Merge || df instanceof Switch || df instanceof BaseTensorOp){
            //Control dependencies and tensor ops (like TensorArray, TensorArrayRead etc) don't need inputs set, execution is a special case
            return sdo;
        }

        //Infer the args based on the inputs (variable + frame + iteration)
        String[] argNames = df.argNames();
        int numArgs = (argNames == null ? 0 : argNames.length);
        int numNonConstIns = (opInputs == null ? 0 : opInputs.size());
        int numNonConstInsAllIters = (allIterInputs == null ? 0 : allIterInputs.size());
        int numConstPhIns = (constAndPhInputs == null ? 0 : constAndPhInputs.size());

        if(numArgs != (numNonConstIns + numConstPhIns + numNonConstInsAllIters)){
            if(numArgs > 1){
                //Might be due to repeated inputs
                Set<String> uniqueArgNames = new HashSet<>();
                Collections.addAll(uniqueArgNames, argNames);
                Preconditions.checkState(uniqueArgNames.size() == (numNonConstIns + numConstPhIns + numNonConstInsAllIters ),
                        "Different number of arg names as op inputs for op %s (%s): arg names %s vs. op inputs %s+%s", df.getClass().getSimpleName(),
                        opName, uniqueArgNames, opInputs, constAndPhInputs);
            } else {
                Preconditions.checkState(numArgs == (numNonConstIns + numConstPhIns ),
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
                } else if(v.getVariableType() == VariableType.VARIABLE){
                    args[i] = v.getArr();
                } else if(v.isPlaceHolder()) {
                    Preconditions.checkState(placeholderValues != null && placeholderValues.containsKey(s), "No array provided for placeholder %s", s);
                    args[i] = placeholderValues.get(s);
                } else {
                    VarId vid = lookup(s, opInputs, allIterInputs, true);
                    args[i] = nodeOutputs.get(vid);
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

            if(df instanceof Identity){
                //We don't need to allocate an output array for Identity, we pass through the input array without copying
                return sdo;
            }

            df.resolvePropertiesFromSameDiffBeforeExecution();  //TODO This is to be removed
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

                if(currOutput == null || currOutput.wasClosed() || !currOutput.shapeDescriptor().equals(reqShape) || currOutput.isEmpty() != reqShape.isEmpty() || isLoop){
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
                    z = mmgr.allocate(false, op.x().dataType(), op.x().shape());
                    op.setZ(z);
                }
            } else {
                List<LongShapeDescriptor> outputShape = ((BaseOp) op).calculateOutputShape();
                Preconditions.checkState(outputShape != null && outputShape.size() == 1, "Could not calculate output shape for op: %s", op.getClass());
                INDArray z = op.z();
                if (z == null || z.wasClosed() || !outputShape.get(0).equals(z.shapeDescriptor()) || isLoop) {
                    if (log.isTraceEnabled()) {
                        log.trace("Existing op result (z) array shape for op {} was {}, allocating new array of shape {}",
                                op.getClass().getSimpleName(), (z == null ? null : Arrays.toString(z.shape())), outputShape.get(0).toString());
                    }

                    LongShapeDescriptor lsd = outputShape.get(0);

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
            VarId inVarId = lookup(n, opInputs, allIterInputs, false);
            Preconditions.checkState(inVarId != null,"Could not find array for variable %s", sdv.getVarName());
            return nodeOutputs.get(inVarId);
        }
    }

    @Override
    protected void onFrameIterTransition(String fromFrame, int fromIter, FrameIter parentFrom, String toFrame, int toIter, FrameIter parentTo){
//        log.info("InferenceSession2: Transition from {}, iter={} (parent={}) to {}, iter={} (parent={})", fromFrame, fromIter, parentFrom, toFrame, toIter, parentTo);
//        //Dependencies for enter ops: these depend on the return to the parent frame - only that that point can we be sure
//        // that those arrays are no longer needed
//        if(!fromFrame.equals(toFrame) && parentFrom != null && parentFrom.getFrame().equals(toFrame)){
//            //This is a transition from a given frame back to its parent frame - i.e., exit current frame
//            Dep d = new ReturnToFrameDep(toFrame, parentTo);
//            arrayUseTracker.markSatisfied(d, true);
//        }
    }

    @Data
    public abstract static class Dep {
        protected String frame;
        protected FrameIter parentFrame;
    }

    @AllArgsConstructor
    @Data
    @EqualsAndHashCode(callSuper = true)
    public static class OpDep extends Dep {
        protected String opName;
        protected int iter;

        protected OpDep(@NonNull String opName, @NonNull String frame, int iter, FrameIter parentFrame){
            this.opName = opName;
            this.frame = frame;
            this.iter = iter;
            this.parentFrame = parentFrame;
        }

        @Override
        public String toString(){
            return "OpDep(" + opName + ",frame=" + frame + ",iter=" + iter + (parentFrame == null ? "" : ",parent=" + parentFrame) + ")";
        }
    }

    /**
     * Represents any frame transition, where X -> Y, where Y == X.parent
     * This is used for deallocating arrays used in enters - which may be needed for the entire duration
     * of an inner frame - and should only be closed when we exit that frame.
     * Now, *usually* frames are nested, but occasionally (mainly TF import) we will have execution in different
     * frames like: main, X, Y, main, where there are enters for both of (main->X) and (main->Y)
     */
    @Data
    @EqualsAndHashCode(callSuper = true)
    public static class ReturnToFrameDep extends Dep {
        public ReturnToFrameDep(@NonNull String frame, FrameIter parentFrame){
            this.frame = frame;
            this.parentFrame = parentFrame;
        }

        @Override
        public String toString(){
            return "ReturnToFrameDep(" + this.frame + (parentFrame == null ? "" : ",parent=" + parentFrame) + ")";
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @AllArgsConstructor
    public static class PlaceholderDep extends Dep {
        protected String phName;
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @AllArgsConstructor
    public static class VariableDep extends Dep {
        protected String varName;
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @AllArgsConstructor
    public static class ConstantDep extends Dep {
        protected String constName;
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @AllArgsConstructor
    public static class ReqOutputDep extends Dep {
        protected String outputName;
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @NoArgsConstructor
    public static class ExecDoneDep extends Dep {

    }
}
