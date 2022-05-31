/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.internal;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.ExecutionResult;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.imports.VariableUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.custom.Invoke;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.CreateView;
import org.nd4j.linalg.api.ops.impl.shape.Stack;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.*;
import org.nd4j.linalg.api.ops.impl.transforms.Assert;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Assign;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.shade.wstx.util.StringUtil;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
public class InferenceSession extends AbstractSession<INDArray, Pair<SameDiffOp,OpContext>> {
    private static final String SCOPE_PANIC_MSG = "If required, arrays in workspaces can be detached using INDArray.detach() before being passed to the SameDiff instance.\n" +
            "Alternatively, arrays defined in a workspace must be replaced after the workspace has been closed.";

    protected static final String KERAS_TRAIN_TEST = "keras_learning_phase";
    //freed array ids to track for allocation, sometimes SDValues contain dup arrays that get freed twice.
    //we track the ids to avoid double frees
    protected Set<Long> freedArrays = new LinkedHashSet<>();

    @Getter
    @Setter
    private SessionMemMgr mmgr;     //Used for allocating and deallocating memory
    /**
     * Array use tracker: What needs to happen before the array can be closed/released?
     * As the name suggests, the INDArrays are tracked using object identity, not equality
     */
    @Getter
    @Setter
    private AbstractDependencyTracker<SDValue, Dep> arrayUseTracker = new IdentityDependencyTracker<>();


    private Map<String,OpContext> opContexts = new HashMap<>();

    public InferenceSession(@NonNull SameDiff sameDiff) {
        super(sameDiff);
        mmgr = new ArrayCacheMemoryMgr();
    }

    @Override
    protected Map<String, INDArray> preprocessPlaceholders(Map<String, INDArray> placeholders, At at) {
        arrayUseTracker.clear();

        //We'll also use this method as a "pre execution" hook-in, to mark variables as something we should never deallocate
        //This occurs by never marking these "ConstantDep" and "VariableDep" instances as satisfied, so there's always
        // an unsatisfied dependency for them in the array use tracker
        //TODO we shouldn't be clearing this on every single iteration, in 99.5% of cases variables will be same as last iteration...
        for (SDVariable v : sameDiff.variables()) {
            if (v.getVariableType() == VariableType.CONSTANT) {
                arrayUseTracker.addDependency(SDValue.create(v.getArr()), new ConstantDep(v.name()));
            } else if (v.getVariableType() == VariableType.VARIABLE) {
                arrayUseTracker.addDependency(SDValue.create(v.getArr()), new VariableDep(v.name()));
            }
        }

        //Workaround for some TF/Keras based models that require explicit train/test as a placeholder
        boolean kerasWorkaround = false;
        List<String> phs = sameDiff.inputs();
        if (phs != null && !phs.isEmpty()) {
            for (String s : phs) {
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


        if (placeholders == null || placeholders.isEmpty()) {
            return placeholders;
        }

        //Handle casting of the input array automatically.
        //The idea here is to avoid unexpected errors if the user (for example) tries to perform inference with a double
        // array for a float placeholder
        //TODO eventually we might have ops that support multiple input types, and hence won't need this casting
        Map<String, INDArray> out = new HashMap<>();
        for (Map.Entry<String, INDArray> e : placeholders.entrySet()) {
            Preconditions.checkState(sameDiff.hasVariable(e.getKey()), "Invalid placeholder passed for execution: " +
                    "No variable/placeholder with name %s exists", e.getKey());
            INDArray arr = e.getValue();
            SDValue arrValue = SDValue.create(arr);
            //First: check workspaces
            if (arr.isAttached()) {
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
            if (kerasWorkaround && e.getKey().endsWith(KERAS_TRAIN_TEST)) {
                arrayUseTracker.addDependency(arrValue, new ExecDoneDep());
            } else if (arr.dataType() == dt) {
                //Mark as a placeholder array in the array use tracker, so we never deallocate this array...
                arrayUseTracker.addDependency(arrValue, new PlaceholderDep(e.getKey()));
            } else {
                INDArray cast = mmgr.allocate(false, dt, arr.shape());
                cast.assign(arr);
                arr = cast;
                //This array CAN be deallocated once consumed, because of the cast
                //TODO we can likely close this sooner
                arrayUseTracker.addDependency(arrValue, new ExecDoneDep());
            }
            out.put(e.getKey(), arr);
        }

        return out;
    }

    @Override
    protected Map<String, SDValue> postProcessOutputValues(Map<String, SDValue> output) {
        //For any queued (not yet processed) ops - mark them as satisfied, so we can deallocate any arrays
        // that are waiting on them
        if (dt.hasNewAllSatisfied()) {
            List<ExecStep> execSteps = dt.getNewAllSatisfiedList();
            for (ExecStep es : execSteps) {
                if (es.getType() == ExecType.OP) {
                    OpDep od = new OpDep(es.getName(), es.getFrameIter().getFrame(), es.getFrameIter().getIteration(), es.getFrameIter().getParentFrame());
                    arrayUseTracker.markSatisfied(od, true);
                }
            }
        }

        //Also mark "end of execution" for array dependency tracker. Mainly used for TensorArray arrays at present.
        //TODO Optimize for reduced memory for some TensorArray operations - i.e., close/deallocate earlier
        arrayUseTracker.markSatisfied(new ExecDoneDep(), true);
        if (arrayUseTracker.hasNewAllSatisfied()) {
            List<SDValue> l = arrayUseTracker.getNewAllSatisfiedList();
            for (SDValue value : l) {
                switch(value.getSdValueType()) {
                    case LIST:
                        for(INDArray arr : value.getListValue())
                            if(arr != null && !freedArrays.contains(arr.getId())) {
                                mmgr.release(arr);
                                freedArrays.add(arr.getId());
                            }
                        break;
                    case TENSOR:
                        if(!freedArrays.contains(value.getTensorValue().getId())) {
                            mmgr.release(value.getTensorValue());
                            freedArrays.add(value.getTensorValue().getId());
                        }
                        break;
                }
            }
        }

        return output;
    }

    @Override
    protected Map<String, INDArray> postProcessOutput(Map<String, INDArray> output) {
        return output;
    }

    @Override
    public ExecutionResult getOutputs(Pair<SameDiffOp, OpContext> opPair,
                                      FrameIter outputFrameIter,
                                      Set<VarId> opInputs,
                                      Set<VarId> allIterInputs,
                                      Set<String> constAndPhInputs,
                                      List<Listener> listeners,
                                      At at, MultiDataSet batch,
                                      Set<String> allReqVariables,
                                      Map<String, SDValue> otherPlaceHolders) {
        SameDiffOp op = opPair.getFirst();
        at.setFrameIter(outputFrameIter);
        if (listeners != null && listeners.size() > 0) {
            SameDiffOp sdOp = sameDiff.getOps().get(op.getOp().getOwnName());
            for (Listener l : listeners) {
                if (l.isActive(at.operation()))
                    l.preOpExecution(sameDiff, at, sdOp, opPair.getSecond());
            }
        }

        if(sameDiff.isDebugMode()) {
            log.info("Executing samediff op: " + op.getName());
        }

        ExecutionResult out = doExec(
                op.getOp(),
                opPair.getRight(),
                outputFrameIter, opInputs,
                allIterInputs,
                constAndPhInputs,
                otherPlaceHolders);
        List<String> opOutNames = op.getOutputsOfOp();

        if (log.isTraceEnabled()) {
            StringBuilder sb = new StringBuilder();
            sb.append(op.getName()).append(" - ").append(outputFrameIter).append(" outputs: ");
            for (int i = 0; i < out.numResults(); i++) {
                if (i > 0)
                    sb.append(", ");
                if(out.hasSingle())
                    sb.append("(").append(i).append(" - ").append(opOutNames.get(i)).append(" = ").append(
                            out.resultAt(i) == null ? null :  out.resultAt(i) .getId()).append(")");

                else if(out.hasValues()) {
                    SDValue value = out.valueWithKeyAtIndex(i, false);
                    //append either the list of associated array ids or the singular one similar to the singular array case
                    String append = value != null && value.getSdValueType() == SDValueType.LIST ? StringUtil.concatEntries(value.getListValue().stream()
                            .map(input -> input.getId()).collect(Collectors.toList()),",",",") : value != null ? String.valueOf(value.getTensorValue().getId()) : null;
                    sb.append("(").append(i).append(" - ").append(opOutNames.get(i)).append(" = ").append(
                            value == null ? null : append).append(")");

                }
            }
            log.trace(sb.toString());
        }

        //Call listeners, before we (maybe) deallocate input arrays
        if (listeners != null && listeners.size() > 0) {
            Map<String, INDArray> namedOuts = null;

            for (Listener l : listeners) {
                if (l.isActive(at.operation())) {
                    //Lazily create map, only if required
                    if (namedOuts == null) {
                        Map<String, INDArray> namedOutsBuilder = new HashMap<>();

                        for (int i = 0; i < out.numResults(); i++)
                            namedOutsBuilder.put(op.outputsOfOp.get(i), out.resultAt(i));
                        namedOuts = Collections.unmodifiableMap(namedOutsBuilder);
                    }


                    l.opExecution(sameDiff, at, batch, op, opPair.getSecond(), out.outputsToArray(opOutNames));

                    for (String varName : namedOuts.keySet()) {
                        l.activationAvailable(sameDiff, at, batch, op, varName, namedOuts.get(varName));
                    }
                }
            }
        }
        op.getOp().clearArrays();
        if(opPair.getSecond() != null)
            opPair.getSecond().purge();


        //Record array uses for memory management/deallocation
        SameDiffOp o = sameDiff.getOps().get(op.getName());
        List<String> outVarNames = o.getOutputsOfOp();
        for (int i = 0; i < out.numResults(); i++) {
            if (out.hasSingle() && out.resultAt(i) == null   || out.hasValues()
                    && out.valueWithKeyAtIndex(i, false) == null && o.getOp() instanceof Switch)
                continue;   //Switch case: we only ever get one of 2 outputs, other is null (branch not executed)
            String name = outVarNames.get(i);
            Variable v = sameDiff.getVariables().get(name);
            List<String> inputsForOps = v.getInputsForOp();
            if (inputsForOps != null) {
                for (String opName : inputsForOps) {
                    //Only add dependencies if we actually need the op this feeds into, otherwise the dependency
                    // will never be marked as satisfied
                    if (!subgraphOps.contains(opName))
                        continue;

                    SameDiffOp forOp = sameDiff.getOps().get(opName);

                    //TODO do switch or merge need special handling also?
                    if (forOp.getOp() instanceof Enter) {
                        Enter e = (Enter) forOp.getOp();
                        if (e.isConstant()) {
                        /*
                        Constant enter case: Need to keep this array around for the entire duration of the frame, including
                        any nested frames, and all iterations.
                        Unfortunately, we don't know exactly when we're done with a frame for good
                        This isn't a great solution, but other possibilities (frame close, trying to detect all exit ops,
                        detecting return to parent frame, etc all fail in certain circumstances, such as due to control dependencies
                        on variables).
                         */
                            Dep d = new ExecDoneDep();
                            addToArrayTracker(out,i,d);
                        } else {
                            Dep d = new OpDep(opName, e.getFrameName(), 0, outputFrameIter);
                            addToArrayTracker(out,i,d);
                        }
                    } else if (forOp.getOp() instanceof NextIteration) {
                        //The array is needed by the NEXT iteration op, not the current one
                        Dep d = new OpDep(opName, outputFrameIter.getFrame(), outputFrameIter.getIteration() + 1, outputFrameIter.getParentFrame());
                        addToArrayTracker(out,i,d);
                    } else if (forOp.getOp() instanceof Exit) {
                        //The array is needed at the EXIT frame (i.e., parent frame), not the inner/just executed one
                        FrameIter fi = outputFrameIter.getParentFrame();
                        Dep d = new OpDep(opName, fi.getFrame(), fi.getIteration(), fi.getParentFrame());
                        addToArrayTracker(out,i,d);
                    } else {
                        //All other ops...
                        Dep d = new OpDep(opName, outputFrameIter.getFrame(), outputFrameIter.getIteration(), outputFrameIter.getParentFrame());
                        addToArrayTracker(out,i,d);
                    }
                }
            }

            if (OUTER_FRAME.equals(outputFrameIter.getFrame()) && allReqVariables.contains(name)) {
                //This variable is an output, record that in the array use tracker, so we don't deallocate it
                //the specific value here
                addToArrayTracker(out,i,new ReqOutputDep(name));
            } else if ((inputsForOps == null || inputsForOps.isEmpty()) && out.getValueOutputs() != null && !arrayUseTracker.hasDependency(out.valueWithKeyAtIndex(i,false))) {
                //This particular array is not actually needed anywhere, so we can deallocate in immediately
                //Possibly only a control dependency, or only one of the outputs of a multi-output op is used
                SDValue array = out.valueWithKeyAtIndex(i, false);
                if (log.isTraceEnabled()) {
                    if(array != null && array.getTensorValue() != null)
                        log.trace("Found array id {} (output of {}) not required anywhere, deallocating", array.getTensorValue().getId(), o.getName());
                }

                if(array != null && array.getTensorValue() != null && !freedArrays.contains(array.getTensorValue().getId())) {
                    mmgr.release(array.getTensorValue());
                    freedArrays.add(array.getTensorValue().getId());
                }
            } else if ((inputsForOps == null || inputsForOps.isEmpty()) && out.getOutputs() != null && !arrayUseTracker.hasDependency(SDValue.create(out.resultAt(i)))) {
                //This particular array is not actually needed anywhere, so we can deallocate in immediately
                //Possibly only a control dependency, or only one of the outputs of a multi-output op is used
                INDArray array = out.resultAt(i);
                if (log.isTraceEnabled()) {
                    if(array != null && array != null)
                        log.trace("Found array id {} (output of {}) not required anywhere, deallocating", array.getId(), o.getName());
                }

                if(array != null && !freedArrays.contains(array.getId())) {
                    mmgr.release(array);
                    freedArrays.add(array.getId());
                }
            }
        }

        //Mark current op dependency as satisfied...
        Dep d = new OpDep(op.getName(), outputFrameIter.getFrame(), outputFrameIter.getIteration(), outputFrameIter.getParentFrame());
        arrayUseTracker.markSatisfied(d, true);


        //Close any no longer required arrays
        if (arrayUseTracker.hasNewAllSatisfied()) {
            List<SDValue> canClose = arrayUseTracker.getNewAllSatisfiedList();
            for (SDValue value : canClose) {
                if (log.isTraceEnabled()) {
                    if(value.getSdValueType() == SDValueType.TENSOR) {
                        INDArray arr = value.getTensorValue();
                        log.trace("Closing array... id={}, {}", arr.getId(), arr.shapeInfoToString());

                    }
                }

                if(!(op.getOp() instanceof Switch))
                    switch(value.getSdValueType()) {
                        case TENSOR:
                            if(!freedArrays.contains(value.getTensorValue().getId())) {
                                mmgr.release(value.getTensorValue());
                                freedArrays.add(value.getTensorValue().getId());
                            }
                            break;
                        case LIST:
                            for(INDArray arr : value.getListValue())
                                if(arr != null && !freedArrays.contains(arr.getId())) {
                                    mmgr.release(arr);
                                    freedArrays.add(arr.getId());
                                }
                            break;
                    }

            }
        }

        return out;
    }


    private void addToArrayTracker(ExecutionResult out,int i,Dep d) {
        if(out.hasSingle()) {
            arrayUseTracker.addDependency(SDValue.create(out.resultOrValueAt(i,false)), d);       //Op defined by "d" needs to be executed before specified array can be closed
        } else {
            arrayUseTracker.addDependency(out.valueWithKeyAtIndex(i,false),d);
        }
    }

    public ExecutionResult doExec(DifferentialFunction op,
                                  OpContext opContext,
                                  FrameIter outputFrameIter,
                                  Set<VarId> opInputs, Set<VarId> allIterInputs,
                                  Set<String> constAndPhInputs,
                                  Map<String, SDValue> otherPlaceHolders) {

        int totalInputs = (opInputs == null ? 0 : opInputs.size()) + (constAndPhInputs == null ? 0 : constAndPhInputs.size())
                + (allIterInputs == null ? 0 : allIterInputs.size());

        boolean constPhInput = (opInputs == null || opInputs.size() == 0) && (allIterInputs == null || allIterInputs.size() == 0);

        if (op instanceof Identity) {
            Identity i = (Identity) op;
            String[] argNames = i.argNames();
            Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in identity op, got %s", (Object) argNames);
            VarId vid = outputFrameIter.toVarId(argNames[0]);
            SDValue orig = getSdValue(vid);
            return ExecutionResult.createValue(vid.getVariable(),orig);
        } else if (op instanceof Switch) {
            Switch s = (Switch) op;
            String[] argNames = s.argNames();       //Order: input, boolean array
            VarId vidPredicate = outputFrameIter.toVarId(argNames[1]);
            SDValue sdValuePred = getSdValue(vidPredicate);
            INDArray predicate = sdValuePred.getTensorValue();
            if(predicate != null && predicate.isEmpty()) {
                predicate = Nd4j.scalar(false);
            }
            if(predicate == null && !constAndPhInputs.isEmpty() && constAndPhInputs.contains(argNames[1])) {
                //Constant predicate...
                predicate = getTensorFromOutputs(new VarId(argNames[1], OUTER_FRAME, 0, null));
            }
            Preconditions.checkNotNull(predicate, "Error during graph execution: Predicate array was null. VarId=%s", vidPredicate);
            Preconditions.checkState(predicate.isScalar() && predicate.dataType() == DataType.BOOL, "Expected boolean predicate: got %ndSInfo", predicate);
            VarId vid = outputFrameIter.toVarId(argNames[0]);
            SDValue sdValue = getSdValue(vid);
            Map<String,SDValue> values = new LinkedHashMap<>();
            ExecutionResult.ExecutionResultBuilder executionResultBuilder = ExecutionResult.builder()
                    .valueOutputs(values);

            if (predicate.getDouble(0) == 0.0) {
                //tensorflow import case
                if(vid.getVariable().equals(vidPredicate.getVariable())) {
                    values.put(vidPredicate.getVariable(),SDValue.create(Arrays.asList(sdValue.getTensorValue(),null)));
                } else {
                    values.put(vid.getVariable(),sdValue);
                    values.put(vidPredicate.getVariable(),null);
                }


            } else {
                //tensorflow import case
                if(vid.getVariable().equals(vidPredicate.getVariable())) {
                    values.put(vidPredicate.getVariable(),SDValue.create(Arrays.asList(null,sdValue.getTensorValue())));
                } else {
                    values.put(vid.getVariable(),null);
                    values.put(vidPredicate.getVariable(),sdValue);
                }


            }

            return executionResultBuilder.build();


        } else if (op instanceof Enter) {
            //Enter op: forwards input to specified execution frame
            Enter e = (Enter) op;
            String[] input = e.argNames();
            Preconditions.checkState(input.length == 1, "Expected only 1 arg name for enter op: got %s", (Object) input);
            Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input for Enter op \"%s\", got %s+%s", e.getOwnName(), opInputs, constAndPhInputs);

            VarId inputVarId;
            if (constPhInput) {
                //Constant or placeholder
                inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0, null);
            } else if (allIterInputs != null && allIterInputs.size() > 0) {
                inputVarId = allIterInputs.iterator().next();
            } else {
                inputVarId = opInputs.iterator().next();
            }

            //note: we strip suffixes on purpose here. DO NOT REMOVE
            inputVarId.setVariable(VariableUtils.stripVarSuffix(inputVarId.getVariable()));

            if(nodeValueOutputs.containsKey(inputVarId)) {
                SDValue value = getSdValue(inputVarId);
                if(value != null && value.getSdValueType() == SDValueType.LIST) {
                    return ExecutionResult.createValue(inputVarId.getVariable(),
                            value);
                } else if(value != null &&  value.getSdValueType() == SDValueType.TENSOR) {
                    INDArray inArr = getTensorFromOutputs(inputVarId);
                    if (inArr == null) {
                        Preconditions.throwStateEx("Could not find array for NextIteration operation %s with output %s (frame=%s, iteration=%s)",
                                op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
                    }

                    return ExecutionResult.createFrom(Arrays.asList(inputVarId.getVariable()),new INDArray[]{inArr});
                } else {
                    throw new IllegalStateException("Illegal value type " + value.getSdValueType() + " for input " + inputVarId);
                }
            } else {
                INDArray inArr = getTensorFromOutputs(inputVarId);
                if (inArr == null) {
                    Preconditions.throwStateEx("Could not find array for Enter operation %s with output %s (frame=%s, iteration=%s)",
                            op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
                }
                return ExecutionResult.createFrom(Arrays.asList(inputVarId.getVariable()),new INDArray[]{inArr});
            }

        } else if (op instanceof Exit) {
            //Exit node forwards input to parent frame

            VarId inputVarId;
            if (constPhInput) {
                //Constant or placeholder
                inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0, null);
            } else if (allIterInputs != null && allIterInputs.size() > 0) {
                inputVarId = allIterInputs.iterator().next();
            } else {
                inputVarId = opInputs.iterator().next();
            }
            SDValue sdValue = getSdValue(inputVarId);
            return ExecutionResult.createValue(inputVarId.getVariable(), sdValue);
        } else if (op instanceof NextIteration) {
            //NextIteration op: forwards its single input to the output of the current frame, but increments the iteration number
            Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input for NextIteration: got %s+%s", opInputs, constAndPhInputs);
            VarId in = (allIterInputs != null && !allIterInputs.isEmpty() ? allIterInputs.iterator().next() : opInputs.iterator().next());
            Preconditions.checkState(outputFrameIter.getFrame().equals(in.getFrame()), "Expected same frame for NextIteration input vs. output:" +
                    " got input %s, output %s", in, outputFrameIter);
            Preconditions.checkState(outputFrameIter.getIteration() == in.getIteration() + 1, "Expected output iteration for NextIteration output to" +
                    " be 1 larger than the input iteration. Input: %s, output %s", in, outputFrameIter);

            if(nodeValueOutputs.containsKey(in) && getSdValue(in) != null) {
                SDValue value = getSdValue(in);
                if(value != null && value.getSdValueType() == SDValueType.LIST) {
                    return ExecutionResult.createValue(in.getVariable(),value);
                } else if(value != null && value.getSdValueType() == SDValueType.TENSOR) {
                    INDArray inArr = getTensorFromOutputs(in);
                    if (inArr == null) {
                        Preconditions.throwStateEx("Could not find array for NextIteration operation %s with output %s (frame=%s, iteration=%s)",
                                op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
                    }

                    return ExecutionResult.createFrom(Arrays.asList(in.getVariable()),new INDArray[]{inArr});
                } else {
                    throw new IllegalStateException("Illegal value type " + value.getSdValueType() + " for input " + in);
                }
            } else {
                INDArray inArr = getTensorFromOutputs(in);
                if (inArr == null) {
                    Preconditions.throwStateEx("Could not find array for NextIteration operation %s with output %s (frame=%s, iteration=%s)",
                            op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
                }
                return ExecutionResult.createFrom(Arrays.asList(in.getVariable()),new INDArray[]{inArr});
            }

        } else if (op instanceof Merge) {
            //Merge available for forward pass when any of its inputs are available. When multiple are available, behaviour
            // is undefined
            Merge m = (Merge) op;
            String[] in = sameDiff.getInputsForOp(op);
            VarId firstInput = outputFrameIter.toVarId(in[0]);
            VarId secondInput = outputFrameIter.toVarId(in[1]);

            SDValue firstValue = getSdValue(firstInput);
            SDValue secondValue = getSdValue(secondInput);
            String s = secondValue != null ? in[1] : in[0];
            VarId vid = secondValue != null ? secondInput :firstInput;
            if(firstValue == null && secondValue == null)
                throw new IllegalStateException("Merge node " + m.getOwnName() + " has no available inputs (all inputs: " + Arrays.toString(in) +
                        ") - should not be executed at this point");
            log.trace("Returning input \"{}\" for merge node \"{}\"", m.getOwnName(), s);
            SDValue value = getSdValue(vid);
            if(value.getSdValueType() == SDValueType.LIST) {
                return ExecutionResult.createValue(vid.getVariable(), getSdValue(vid));
            } else if(value.getSdValueType() == SDValueType.TENSOR) {
                INDArray inArr = getTensorFromOutputs(vid);
                if (inArr == null) {
                    Preconditions.throwStateEx("Could not find array for NextIteration operation %s with output %s (frame=%s, iteration=%s)",
                            op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
                }

                return ExecutionResult.createFrom(Arrays.asList(vid.getVariable()),new INDArray[]{inArr});
            } else {
                throw new IllegalStateException("Illegal value type " + value.getSdValueType() + " for input " + in);
            }



        } else if (op instanceof LoopCond) {
            //LoopCond just forwards scalar boolean to output
            LoopCond lc = (LoopCond) op;
            String[] argNames = lc.argNames();
            Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in LoopCond op, got %s", (Object) argNames);
            VarId vid = outputFrameIter.toVarId(argNames[0]);
            SDValue getValue = getSdValue(vid);
            if(getValue.getTensorValue() == null) {
                throw new IllegalStateException("Node value output at " + vid.getVariable() + " was not a boolean tensor!");
            }
            Preconditions.checkNotNull(getValue, "Input to LoopCond op must not be null");
            Preconditions.checkState(getValue.getTensorValue().isScalar() && getValue.getTensorValue().dataType() == DataType.BOOL, "LoopCond input must be a scalar boolean, got %ndShape");
            return ExecutionResult.createValue(vid.getVariable(), getValue);
        } else if (op instanceof BaseTensorOp) {
            //TensorOps - special cases...
            return getOutputsHelperTensorArrayOps(op, outputFrameIter, opInputs, allIterInputs, otherPlaceHolders);
        } else if(op instanceof Identity) {
            List<VarId> orderedInputs = new ArrayList<>(opInputs);
            SDValue sdValue = getSdValue(orderedInputs.get(0));
            return ExecutionResult.createValue(op.outputVariablesNames()[0], sdValue);

        } else if(op instanceof Assign) {
            List<VarId> orderedInputs = new ArrayList<>(opInputs);
            if(orderedInputs.size() > 1) {
                SDValue sdValue = getSdValue(orderedInputs.get(0));
                SDValue sdValue1 = getSdValue(orderedInputs.get(1));
                switch(sdValue.getSdValueType()) {
                    case TENSOR:
                        Assign c = (Assign) op;
                        Nd4j.exec(c, opContext);
                        return ExecutionResult.createFrom(c,opContext);
                    case LIST:
                        return ExecutionResult.createValue(op.outputVariablesNames()[0], sdValue1);

                }

            }

            SDValue sdValue = getSdValue(orderedInputs.get(0));
            return ExecutionResult.createValue(op.outputVariablesNames()[0], sdValue);

        } else if (op instanceof GradientBackwardsMarker) {
            INDArray out = mmgr.allocate(false, DataType.FLOAT).assign(1.0f);
            return ExecutionResult.createFrom(Arrays.asList("gradientbackwardsmarker"), new INDArray[]{out});
        } else if(op instanceof CreateView) {
            Map<String,VarId> inputVars = new LinkedHashMap<>();
            String[] argNames = op.argNames();
            for(Iterator<VarId> iter = opInputs.iterator(); iter.hasNext();) {
                VarId varId  = iter.next();
                inputVars.put(varId.getVariable(),varId);
            }
            SDValue sdValue = getSdValue(inputVars.get(argNames[0]));
            if(sdValue == null) {
                sdValue = SDValue.create(opContext.getInputArray(0));
            }
            INDArray[] indices = new INDArray[argNames.length - 1];
            for(int i = 1; i < argNames.length; i++) {
                indices[i - 1] = getSdValue(inputVars.get(argNames[i])).getTensorValue();
            }
            return ExecutionResult.createFrom(op.outputVariablesNames()[0],
                    CreateView.createFrom(sdValue.getTensorValue(),indices));
        } else if (op instanceof ExternalErrorsFunction) {
            ExternalErrorsFunction fn = (ExternalErrorsFunction) op;
            String n = fn.getGradPlaceholderName();
            INDArray arr = getTensorFromOutputs(new VarId(n, OUTER_FRAME, 0, null));
            Preconditions.checkState(arr != null, "Could not find external errors placeholder array: %s", arr);
            INDArray out = mmgr.allocate(false, arr.dataType(), arr.shape());
            out.assign(arr);
            return ExecutionResult.createFrom(Arrays.asList(n), new INDArray[]{out});
        } else if(op instanceof Invoke) {
            Invoke invoke = (Invoke) op;
            boolean hasValues = false;
            for(VarId varId : opInputs) {
                //need to invoke with values
                if(nodeValueOutputs.containsKey(varId)) {
                    hasValues = true;
                    break;
                }
            }

            //no need to check placeholders if other values are present
            if(!hasValues)
                for(Map.Entry<String,SDValue> entry : otherPlaceHolders.entrySet()) {
                    if(constAndPhInputs.contains(entry.getKey())) {
                        hasValues = true;
                        break;
                    }
                }

            Map<String,INDArray> inputs = new LinkedHashMap<>();
            Map<String,SDValue> valueInputs = new LinkedHashMap<>();
            //need to pull from tensor arrays
            if(!hasValues) {
                //simple linear scan of inputs over inputs
                int currInput = 0;
                for(VarId opInput : opInputs) {
                    inputs.put(opInput.getVariable(),opContext.getInputArray(currInput));
                    currInput++;
                }
            } else {
                //simple linear scan of inputs over inputs
                Map<String,VarId> varIdsByVariable = new HashMap<>();
                for(VarId opInput : opInputs) {
                    varIdsByVariable.put(opInput.getVariable(),opInput);
                }

                for(int i = 0; i < invoke.getInputVarNames().length; i++) {
                    VarId opInput = varIdsByVariable.get(invoke.getInputVarNames()[i]);
                    if(constAndPhInputs.contains(invoke.getInputVarNames()[i])) {
                        if(otherPlaceHolders.containsKey(invoke.getInputVarNames()[i]))
                            valueInputs.put(invoke.getInputVarNames()[i],otherPlaceHolders.get(invoke.getInputVarNames()[i]));
                        else if(inputs.containsKey(invoke.getInputVarNames()[i]))
                            valueInputs.put(invoke.getInputVarNames()[i],SDValue.create(inputs.get(invoke.getInputVarNames()[i])));
                    }else if(sameDiff.getArrForVarName(invoke.getInputVarNames()[i]) != null) {
                        valueInputs.put(invoke.getInputVarNames()[i],SDValue.create(sameDiff.getArrForVarName(invoke.getInputVarNames()[i])));
                    }  else if(nodeValueOutputs.containsKey(opInput)) {
                        valueInputs.put(opInput.getVariable(), getSdValue(opInput));
                    } else {
                        valueInputs.put(opInput.getVariable(),SDValue.create(opContext.getInputArray(i)));
                    }
                }
            }

            if(valueInputs.size() + inputs.size() != op.args().length) {
                throw new IllegalArgumentException("Value inputs and inputs combined did not fulfill all arguments. Inputs were: " + Arrays.toString(op.argNames()) + " for op name " + op.getOwnName());
            }


            return Invoke.doInvoke(invoke,inputs,valueInputs);
        } else if (op instanceof Assert) {
            Assert a = (Assert)op;
            boolean condition = opContext.getInputArray(0).getDouble(0) != 0.0;
            if(!condition) {
                //Assertion failed
                String s = "Assertion failed for operation \"" + op.getOwnName() + "\" during execution";
                if(a.numInputArguments() >= 3) {
                    INDArray msg = opContext.getInputArray(2);
                    if (msg != null && msg.dataType() == DataType.UTF8) {
                        s += ": " + msg.getString(0);
                    }
                }
                if(a.numInputArguments() >= 5) {
                    INDArray arr = opContext.getInputArray(4);
                    s += "\n" + arr;
                }
                throw new IllegalStateException(s);
            }
            return ExecutionResult.createFrom(a,opContext);
        } else if (op instanceof CustomOp) {
            CustomOp c = (CustomOp) op;
            Nd4j.exec(c, opContext);
            return ExecutionResult.createFrom((DifferentialFunction) c,opContext);
        } else if (op instanceof Op) {
            Op o = (Op) op;
            Nd4j.exec(o, opContext);
            return ExecutionResult.createFrom((DifferentialFunction)o,opContext);
        } else {
            throw new UnsupportedOperationException("Execution not yet implemented for: " + op.getClass().getName());
        }
    }

    private SDValue getPreviousValue(VarId varId) {
        return getPreviousValue(varId,1);
    }

    private SDValue getPreviousValue(VarId varId,int offset) {
        VarId ret = new VarId(varId.getVariable(), varId.getFrame(), varId.getIteration() - offset,varId.getParentFrame());
        return nodeValueOutputs.get(ret);
    }

    private SDValue getValueAtIteration(String var,String frame, int iteration,FrameIter parentFrame) {
        VarId varId = new VarId(var,frame,iteration,parentFrame);
        return nodeValueOutputs.get(varId);
    }

    /**
     * Forward pass for TensorArray ops
     */
    public ExecutionResult getOutputsHelperTensorArrayOps(DifferentialFunction op, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs, Map<String, SDValue> otherPlaceHolders) {
        /*
        TODO: TensorArray memory management note: For now, we'll close any INDArrays stored in the TensorArray at the end of
        graph execution. This uses more memory than necessary for an earlier close strategy, but simplifies memory management.
        This should be revisited and optimized later
         */

        if (op instanceof TensorArray) {
            //Create a TensorArray
            /**
             * TODO: take new conversion of static shape attribute to new input
             * and ensure that it's a proper argument in the tensor array arg.
             * Also enforce it in tensorarrayread.
             *
             * Remember, the bug for this is coming from the first element in the shape
             * being a 1 x 3 while the rest is 1 x5. The goal here is to enforce
             * stricter standards and be able to verify what elements should be
             * what shape.
             *
             * Beyond that, we still need to find the bug for what causes the 1 x 3
             * in the first tensorarraywrite.
             */
            VarId vid = outputFrameIter.toVarId(op.outputVariable().name());
            Preconditions.checkState(!nodeValueOutputs.containsKey(vid), "TensorArray already exists for %s when executing TensorArrayV3", vid);
            SDVariable size = op.arg(0);
            INDArray arr = size.getArr();
            List<INDArray> createList = new ArrayList<>();
            for(int i = 0; i  < arr.getInt(0); i++) {
                createList.add(null);
            }

            SDValue listValue = SDValue.create(createList);
            putNodeValue(listValue, vid);

            /**
             * TODO:  add support for loading size of list (add a bunch of nulls if needed to match size)
             * from the input arguments. Right now adding a new list as empty is actually a bug.
             *
             * TODO: check TF import to make sure we support this use case.
             */

            // Note that TensorArray has 2 outputs - a 'dummy' SDVariable that represents it, and a second output (return a scalar 0.0)
            return ExecutionResult.createValue(vid.getVariable(),listValue);
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
            VarId v = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (v == null && allIterInputs != null) {
                v = lookup(inTensorArray.name(), allIterInputs, false);
            }


            Preconditions.checkState(v != null, "Could not find input %s", inTensorArray.name());

            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayRead
                //TODO also TensorArrayWrite, scatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                v = v.getParentFrame().toVarId(inTensorArray.name());
            }

            TensorArray tensorArray1 = TensorArray.getTensorArray(sameDiff, inTensorArray);

            List<INDArray> list = null;
            if(!nodeValueOutputs.containsKey(v)) {
                TensorArray tensorArray = TensorArray.getTensorArray(sameDiff,inTensorArray);
                SDVariable output = tensorArray.getVar();
                list = getTensorArraysInSession(output.name());

            } else {
                list = getSdValue(v).getListValue();
            }

            //we specify a shape every element should be and validate it
            if(tensorArray1.args().length > 1) {
                INDArray inputShape = tensorArray1.arg(1).getArr();
                long[] inputShapeArr = inputShape.toLongVector();
                for(int j = 0; j < list.size(); j++) {
                    if(list.get(j) != null)
                        if(!Arrays.equals(inputShape.toLongVector(),list.get(j).shape())) {
                            throw new IllegalArgumentException("Element " + j  + " of list " + v.getVariable() + " did not have correct shape of " + Arrays.toString(inputShapeArr) + " was shape " + Arrays.toString(list.get(j).shape()));
                        }

                }
            }
            Preconditions.checkState(list != null, "Could not find TensorList for %s", v);
            Preconditions.checkState(list.size() > i, "Cannot get index %s from TensorList of size %s (array not present?) - VarId=%s", i, list.size(), v);

            INDArray out = list.get(i);

            System.out.println("Reading item at index " + i + " for list " + v + " with value " + out + " with list of " + list);
            return ExecutionResult.createFrom(v.getVariable(),out);
        } else if (op instanceof TensorArrayWrite) {
            //TensorArrayWrite - also has a scalar 0.0 that it returns...
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            //Work out the varid (frame/iteration) of the tensor array:
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }

            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }


            //create new tensor array for placeholder referencing a passed in variable
            if(tArr == null && inTensorArray.getVariableType() == VariableType.PLACEHOLDER) {
                VarId varId = new VarId(inTensorArray.name(),outputFrameIter.getFrame(),outputFrameIter.getIteration(),outputFrameIter.getParentFrame());
                tArr = varId;
                SDValue sdValue = otherPlaceHolders.get(inTensorArray.name());
                putNodeValue(sdValue, tArr);
            }

            Preconditions.checkState(tArr != null, "Could not find input %s", inTensorArray.name());



            //Input 0 is the TensorArray (or dummy variable that represents it) - but sometimes Enter, in TensorArray -> Enter -> TensorARrayRead
            //Input 1 is the index
            //Input 2 is the value to write

            String idxName = op.arg(1).name();
            SDVariable idxSDV = sameDiff.getVariable(idxName);
            INDArray idxArr = getArray(idxSDV, opInputs, allIterInputs);
            Preconditions.checkState(idxArr.isScalar(), "Index variable ID for TensorArrayWrite should be a scalar, got %ndShape", idxArr);
            int idx = idxArr.getInt(0);

            String inName = op.arg(2).name();
            SDVariable inSDV = sameDiff.getVariable(inName);
            INDArray arr = getArray(inSDV, opInputs, allIterInputs);
            Preconditions.checkState(arr != null, "Could not find array for %s", inName);
            TensorArray tArrOp = TensorArray.getTensorArray(sameDiff,inTensorArray);
            if(tArrOp.args().length > 1) {
                long[] shape = tArrOp.arg(1).getArr().toLongVector();
                if(!Arrays.equals(arr.shape(),shape)) {
                    throw new IllegalArgumentException("Unable to write array of shape " + Arrays.toString(arr.shape()) + " must be " + shape + " for op " + op.getOwnName() + " and tensor array " + tArrOp.getOwnName());
                }
            }


            Preconditions.checkState(nodeValueOutputs.containsKey(tArr), "Tensor array does not exist for %s", tArr);
            //TODO is this always safe to insert by index for all execution orders?
            SDValue sdValue1 = getSdValue(tArr);
            List<INDArray> l = sdValue1.getListValue(); //.set(idx, arr);
            if(idx < 0 && l != null && !l.isEmpty()) {
                idx += l.size() + 1;
            } else if(idx < 0) {
                idx = 0;
            }
            while (l.size() <= idx) {
                //Can't use set(int, E) if index >= size
                l.add(null);
            }

            setArrayAtIndex(l, idx, arr);
            System.out.println("Setting item at index " + idx + " for list " + tArr + " with value " + arr + " with whole list of after write " + l + " and value array " + arr);
            System.out.println("Writing value " + inSDV + " to list " + tArr.getVariable() + " at iteration " + tArr.getIteration());

            //Add a dependency
            Dep d = new ExecDoneDep();
            putNodeValue(sdValue1, tArr);
            arrayUseTracker.addDependency(sdValue1, d);
            return ExecutionResult.createValue(op.outputVariable().name(),sdValue1);
        } else if (op instanceof TensorArraySize) {
            //Index 0 is the TensorArray (or dummy variable that represents it)
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            TensorArray tensorArray = TensorArray.getTensorArray(sameDiff,inTensorArray);
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }

            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }

            List<INDArray> l = getSdValue(tArr).getListValue();
            int size = l == null ? 0 : l.size();
            INDArray scalar = mmgr.allocate(false, DataType.INT).assign(size);
            return ExecutionResult.createFrom(tensorArray.getVar().name(),scalar);
        } else if (op instanceof TensorArrayConcat) {
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }
            List<INDArray> l = getSdValue(tArr).getListValue();

            Concat c = new Concat(0, l.stream().filter(input -> input != null).collect(Collectors.toList())
                    .toArray(new INDArray[0]));
            List<LongShapeDescriptor> shape = c.calculateOutputShape();
            INDArray out = mmgr.allocate(false, shape.get(0));
            c.setOutputArgument(0, out);
            Nd4j.exec(c);
            return ExecutionResult.createFrom(tArr.getVariable(),out);
        } else if (op instanceof TensorArrayGather) {
            //Input 0: the TensorArray
            //Input 1: the indices (1d integer vector)

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }

            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }

            List<INDArray> l = getSdValue(tArr).getListValue();
            Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

            String indicesName = op.arg(1).name();
            SDVariable indicesSDV = sameDiff.getVariable(indicesName);
            INDArray idxArr = indicesSDV.getArr();
            Preconditions.checkState(idxArr.isVector(), "Indices variable for TensorArrayGather should be a vector, got %ndShape for %s", idxArr, indicesName);
            Preconditions.checkState(idxArr.dataType().isIntType(), "Indices variable for TensorArrayGather should be an integer type, got %s for array %s", idxArr.dataType(), indicesName);

            int[] idxArrInt = idxArr.toIntVector();
            System.out.println("Gathering op " + op.getOwnName() + " from indices " + Arrays.toString(idxArrInt) + " named " + indicesName + " from list " + tArr.getVariable());
            if(idxArrInt.length > 0) {
                //Edge case: -1 means "all"
                List<INDArray> newList = new ArrayList<>();
                if (idxArrInt.length == 1 || idxArrInt.length > 0 &&  idxArrInt[0]  < 0) {
                    newList.addAll(l);
                } else {
                    for (int id : idxArrInt) {
                        Preconditions.checkState(id >= 0, "Index for TensorArrayGather must be >= 0, got %s", id);
                        if(l.get(id) != null) {
                            System.out.println("Gathering op " + op.getOwnName() + " at index " + id + " adding value " + l.get(id).toStringFull() + " from full list " + l);
                            newList.add(l.get(id));

                        }
                    }
                }

                Stack s = new Stack(newList.stream().filter(input -> input != null).collect(Collectors.toList())
                        .toArray(new INDArray[0]), null, 0);
                List<LongShapeDescriptor> shape = s.calculateOutputShape();
                INDArray out = mmgr.allocate(false, shape.get(0));
                s.setOutputArgument(0, out);
                Nd4j.exec(s);
                return ExecutionResult.createFrom(tArr.getVariable(),out);
            } else {
                return ExecutionResult.createFrom(tArr.getVariable(),Nd4j.zeros(op.arg().dataType(),0));
            }

        } else if (op instanceof TensorArrayScatter) {
            //Scatter values from a rank (N+1)d tensor into specific indices of the TensorArray
            //Input 0: the TensorArray
            //Input 1: the indices (1d integer vector)
            //Input 2: The values to scatter

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            TensorArray ta = (TensorArray) sameDiff.getVariableOutputOp(inTensorArray.name());
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }

            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }

            SDValue retValue = getSdValue(tArr);
            List<INDArray> l = retValue.getListValue();
            Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

            String indicesName = op.arg(1).name();
            SDVariable indicesSDV = sameDiff.getVariable(indicesName);
            INDArray idxArr = indicesSDV.getArr();
            Preconditions.checkState(idxArr.isVector(), "Indices variable for TensorArrayScatter should be a vector, got %ndShape for %s", idxArr, indicesName);
            Preconditions.checkState(idxArr.dataType().isIntType(), "Indices variable for TensorArrayScatter should be an integer type, got %s for array %s", idxArr.dataType(), indicesName);
            int[] idxs = idxArr.toIntVector();

            String valuesName = op.arg(2).name();
            SDVariable valuesSDV = sameDiff.getVariable(valuesName);
            INDArray valuesArr = getArray(valuesSDV, opInputs, allIterInputs);

            while (l.size() < idxs.length) { //Can't use set(int, E) if index >= size
                l.add(null);
            }


            //Edge case: idxs being [-1] means "all sub arrays" (i.e., "unstack" case)
            if (idxs.length == 1 && idxs[0] == -1) {
                idxs = ArrayUtil.range(0, (int) valuesArr.size(0));
            }

            for(int i = 0; i < idxs.length; i++) {
                if(valuesArr.size(0) < idxs[i]) {
                    throw new IllegalArgumentException("Unable to obtain slice from values array named " + valuesName +  " with shape " + Arrays.toString(valuesArr.shape()) + " at index " + idxs[i] + " at node named " + op.getOwnName()  + " with inputs " + Arrays.toString(op.argNames()));
                }
            }

            for (int i = 0; i < idxs.length; i++) {
                if(idxs[i] >= valuesArr.size(0)) {
                    throw new IllegalStateException("Unable to pull slice from value array " + valuesSDV.name() + " of shape " + Arrays.toString(valuesArr.shape()) + " index was" + idxs[i]  + " all indices were " + Arrays.toString(idxs));
                }
                INDArray getView = valuesArr.slice(idxs[i]);
                INDArray get = mmgr.dup(getView);
                if(ta.args().length > 1) {
                    long[] shape = ta.arg(1).getArr().toLongVector();
                    if(!Arrays.equals(get.shape(),shape)) {
                        throw new IllegalArgumentException("Unable to write array of shape " + Arrays.toString(get.shape()) + " must be " + shape + " for op " + op.getOwnName() + " and tensor array " + ta.getOwnName());
                    }
                }
                SDValue newValue = SDValue.create(get);
                int outIdx = idxs[i];
                if (valuesArr.rank() == 1 && get.rank() > 0) {
                    get = get.reshape();
                }

                //reflect the expanded storage
                if(outIdx >= l.size()) {
                    while(l.size() <= outIdx) {
                        l.add(null);
                    }
                }

                System.out.println("Scattering item at index " + i + " for list " + tArr + " with value " + get + " from whole list of " + l + " from values array " + valuesArr.toStringFull() + " named " + valuesSDV.name());
                setArrayAtIndex(l, outIdx, get);

                //Add dependency for values array until end of execution
                arrayUseTracker.addDependency(newValue, new ExecDoneDep());
            }


            return ExecutionResult.createValue(valuesName,retValue);
        } else if (op instanceof TensorArraySplit) {
            //Split values from a rank (N+1)d tensor into sequential indices of the TensorArray
            //For example, orig=[8,2] sizearray with split (4,4) means TensorArray[0] = orig[0:4,:] and TensorArray[1] = orig[4:8,:]
            //Input 0: the TensorArray
            //Input 1: The values to split
            //Input 2: the size of each split (1d integer vector)

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }


            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }

            SDValue sdValue = getSdValue(tArr);
            List<INDArray> l = sdValue.getListValue();
            Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

            String splitName = op.arg(1).name();
            INDArray splitArr = getArray(sameDiff.getVariable(splitName), opInputs, allIterInputs);


            String sizeName = op.arg(2).name();
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
                SDValue subValue = SDValue.create(sub);
                setArrayAtIndex(l, i, sub);
                soFar += sizes[i];

                //Add dependency for values array until end of execution
                arrayUseTracker.addDependency(subValue, new ExecDoneDep());
            }

            return ExecutionResult.createValue(sizeName,sdValue);
        } else if (op instanceof TensorArrayRemove) {
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            SDVariable index = op.arg(1);
            List<INDArray> l = getTensorArraysInSession(inTensorArray.name());
            l.remove(index.getArr(true).getInt(0));
            INDArray scalar = mmgr.allocate(false, DataType.FLOAT).assign(0.0);
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }

            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }

            //setup an extra reference to the removed list
            putNodeValue(SDValue.create(l), tArr);
            return ExecutionResult.createValue(tArr.getVariable(),Arrays.asList(scalar));
        }

        else {
            throw new IllegalStateException("Execution support not yet implemented for: " + op.getClass().getName());
        }
    }


    private Map<Pair<String,Integer>,SDValue> valuesFor(String varName) {
        Map<Pair<String,Integer>,SDValue> ret = new HashMap<>();
        for(Map.Entry<VarId,SDValue> values : nodeValueOutputs.entrySet()) {
            if(values.getKey().getVariable().equals(varName)) {
                ret.put(Pair.of(values.getKey().getVariable(),values.getKey().getIteration()),values.getValue());
            }
        }

        return ret;
    }


    @Override
    public INDArray getConstantOrVariable(String variableName) {
        SDVariable v = sameDiff.getVariable(variableName);
        Preconditions.checkState(sameDiff.getVariable(variableName).isConstant() || v.getVariableType() == VariableType.VARIABLE,
                "Variable %s is not a constant", variableName);
        return sameDiff.getArrForVarName(variableName);
    }

    @Override
    public Pair<SameDiffOp,OpContext> getAndParameterizeOp(String opName, FrameIter frameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                                           Set<String> constAndPhInputs, Map<String, INDArray> placeholderValues, Set<String> allReqVariables, Map<String, SDValue> otherPlaceholders) {
        SameDiffOp sdo = sameDiff.getOps().get(opName);
        DifferentialFunction df = sdo.getOp();

        //TODO Switch to OpContext - and make sure executing like that is thread safe (i.e., array fields in ops are not used etc)

        Preconditions.checkNotNull(df, "No differential function found with name \"%s\"", opName);

        if (df instanceof LoopCond || df instanceof Enter || df instanceof Exit || df instanceof NextIteration ||
                df instanceof Merge || df instanceof Switch || df instanceof BaseTensorOp || df instanceof Invoke) {
            //Control dependencies and tensor ops (like TensorArray, TensorArrayRead etc) don't need inputs set, execution is a special case
            return new Pair<>(sdo, null);
        }

        //Infer the args based on the inputs (variable + frame + iteration)
        String[] argNames = df.argNames();
        int numArgs = (argNames == null ? 0 : argNames.length);
        int numNonConstIns = (opInputs == null ? 0 : opInputs.size());
        int numNonConstInsAllIters = (allIterInputs == null ? 0 : allIterInputs.size());
        int numConstPhIns = (constAndPhInputs == null ? 0 : constAndPhInputs.size());

        if (numArgs != (numNonConstIns + numConstPhIns + numNonConstInsAllIters)) {
            if (numArgs > 1) {
                //Might be due to repeated inputs
                Set<String> uniqueArgNames = new LinkedHashSet<>();
                Collections.addAll(uniqueArgNames, argNames);

            } else {
                Preconditions.checkState(numArgs == (numNonConstIns + numConstPhIns),
                        "Different number of arg names as op inputs for op %s (%s): arg names %s vs. op inputs %s+%s", df.getClass().getSimpleName(),
                        opName, argNames, opInputs, constAndPhInputs);
            }
        }

        INDArray[] args = null;
        if (argNames != null && argNames.length > 0) {
            args = new INDArray[argNames.length];
            int i = 0;
            for (String s : argNames) {
                SDVariable v = sameDiff.getVariable(s);
                if (v.isConstant()) {
                    args[i] = v.getArr();
                } else if (v.getVariableType() == VariableType.VARIABLE) {
                    args[i] = v.getArr();
                } else if (v.isPlaceHolder()) {
                    if(placeholderValues != null && placeholderValues.containsKey(s))
                        args[i] = placeholderValues.get(s);
                    else if(otherPlaceholders != null && otherPlaceholders.containsKey(s)) {
                        args[i] = otherPlaceholders.get(s).getTensorValue();
                    }
                    else
                        throw new IllegalArgumentException("No array was provided for required placeholder variable \"%s\"".format(s));
                } else {
                    VarId vid = lookup(s, opInputs, allIterInputs, true);
                    SDValue getValue = getSdValue(vid);
                    if(getValue != null)
                        switch(getValue.getSdValueType()) {
                            case TENSOR:
                                args[i] = getValue.getTensorValue();
                                break;
                            case LIST:
                                DifferentialFunction variableOutputOp = sameDiff.getVariableOutputOp(s);
                                //tensorflow import case: when switch is imported and 2 are input names are equal
                                //we output a list with 1 value that's null and 1 that's not
                                if(variableOutputOp instanceof Switch && variableOutputOp.argNames().length == 2 && variableOutputOp.argNames()[0].equals(variableOutputOp.argNames()[1])) {
                                    //find the non null value
                                    for(int j = 0; j < getValue.getListValue().size(); j++) {
                                        if(getValue.getListValue().get(j) !=  null) {
                                            args[i] = getValue.getListValue().get(j);
                                            break;
                                        }
                                    }
                                }
                                else
                                    args[i] = Nd4j.empty(DataType.FLOAT);
                                break;

                        }
                }


                Preconditions.checkNotNull(args[i], "Could not parameterize op %s: array %s (variable %s) is null", opName, i, v.name());
                i++;
            }
        }

        //Set the op inputs and output arguments
        //Note that when we are in a loop (and non-first iteration), we want to allocate new arrays even if shapes are
        // ok: this is because we need the values in past iterations for backprop (potentially)
        //TODO let's find a way to use in-place modification for loops where possible to reduce memory requirements
        boolean isLoop = !frameIter.getFrame().equals(OUTER_FRAME) && frameIter.getIteration() > 0;

        OpContext oc = opContexts.get(opName);
        if(oc == null){
            oc = Nd4j.getExecutioner().buildContext();
            opContexts.put(opName, oc);
        }

        if (df instanceof CustomOp) {
            DynamicCustomOp customOp = (DynamicCustomOp) df;
            if (df instanceof Identity || df instanceof CreateView) {
                if (args != null) {
                    oc.setInputArrays(args);
                }
                //We don't need to allocate an output array for Identity, we pass through the input array without copying
                return new Pair<>(sdo, oc);
            }

            oc.setArgs(args, customOp.iArgs(), customOp.dArgs() , customOp.tArgs(), customOp.bArgs() );


            List<LongShapeDescriptor> outShape = customOp.calculateOutputShape(oc);
            Preconditions.checkState(outShape != null && outShape.size() > 0, "Failed to calculate output shapes for op %s (%s) - no shapes were returned by calculateOutputShape()", customOp.opName(), customOp.getOwnName());
            String[] outNames = df.outputVariablesNames();
            Preconditions.checkState(outNames.length == outShape.size(), "Error in operation shape calculation for op \"%s\": Got %s op output shapes for an operation" +
                    " with %s outputs (number of shapes and outputs must be equal)", df.opName(), outShape.size(), outNames.length);
            for (int i = 0; i < outShape.size(); i++) {
                LongShapeDescriptor reqShape = outShape.get(i);

                //Issue: many ops have multiple valid output datatypes, and output shape calc can't at present know which: https://github.com/eclipse/deeplearning4j/issues/6872
                //As a workaround, we'll use the output variable datatype instead.
                DataType dt = sameDiff.getVariable(outNames[i]).dataType();
                DataType currDT = reqShape.dataType();
                if (dt != currDT) {
                    reqShape = reqShape.asDataType(dt);
                }

                //Always allocate new output array, rely on memory manager for efficient memory management and array reuse etc
                boolean isOutput = allReqVariables.contains(outNames[i]);
                INDArray out = mmgr.allocate(isOutput, reqShape);
                if(reqShape.isEmpty() && !out.isEmpty()) {
                    throw new IllegalStateException("Output shape was empty, but created array was not.");
                }

                oc.setOutputArray(i, out);
            }

        } else if (df instanceof Op) {
            Op op = (Op) df;

            boolean axisArg = false;
            boolean emptyReduce = false;
            if (op instanceof ReduceOp && ((ReduceOp) op).getOpType() != Op.Type.REDUCE3 && df.argNames().length == 2) {
                //2nd input should be treated as integer axis arg...
                SDVariable axisArgVar = df.arg(1);
                Preconditions.checkState(axisArgVar.dataType().isIntType(), "Legacy op %s input 1 (axis) was expected to be an integer type, is %s", df.getClass(), axisArgVar.dataType());

                INDArray arr = getArray(axisArgVar, opInputs, allIterInputs);
                Preconditions.checkState(arr != null, "Could not get axis argument for op %s: %s", df.getOwnName(), df.getClass());
                if (!arr.isEmpty()) {
                    int[] axis = arr.toIntVector();
                    int rank = args[0].rank();
                    axis = Shape.normalizeAxis(rank, axis);
                    df.setDimensions(axis);
                    ((BaseReduceOp) op).setEmptyReduce(false);
                } else {
                    df.setDimensions(null);
                    emptyReduce = true;
                    //Note: edge case: [x,y].sum(empty) = [x,y] for TF import compatibility.
                    //Note also that empty is not the same as int[0] as in INDArray.sum(new int[0])
                    ((BaseReduceOp) op).setEmptyReduce(true);
                }
                axisArg = true;
            } else if (op instanceof ScalarOp && df.argNames().length == 2) {
                //Scalar ops: 2nd input should be treated as scalar...
                SDVariable scalarVar = df.arg(1);
                INDArray scalar = getArray(scalarVar, opInputs, allIterInputs);
                Preconditions.checkState(scalar != null, "Could not get scalar argument for op %s: %s", df.getOwnName(), df.getClass());
                Preconditions.checkState(scalar.isScalar(), "Scalar argument for op %s (%s) is not a scalar: has shape %ndShape", df.getOwnName(), df.getClass(), scalar);
                ((ScalarOp) op).setScalar(scalar);
            }

            if (args != null && args.length > 0) {
                oc.setInputArray(0, args[0]);
                if (args.length == 2 && !axisArg)
                    oc.setInputArray(1, args[1]);
            }


            //Check output shape; allocate a new Z if required
            //For example, if minibatch size has changed since last op execution
            boolean isOutput = allReqVariables.contains(((BaseOp) op).outputVariablesNames()[0]);
            if (emptyReduce) {
                //Always allocate new output array, rely on memory manager for efficient memory management and array reuse etc
                INDArray z = mmgr.allocate(false, oc.getInputArray(0).dataType(), oc.getInputArray(0).shape());
                oc.setOutputArray(0, z);
            } else {
                List<LongShapeDescriptor> outputShape = ((BaseOp) op).calculateOutputShape(oc);
                Preconditions.checkState(outputShape != null && outputShape.size() == 1, "Could not calculate output shape for op: %s", op.getClass());
                LongShapeDescriptor lsd = outputShape.get(0);
                INDArray z = mmgr.allocate(isOutput, lsd);
                oc.setOutputArray(0, z);
            }
        }

        return new Pair<>(sdo, oc);
    }


    protected INDArray getArray(SDVariable sdv, Collection<VarId> opInputs, Collection<VarId> allIterInputs) {
        String n = sdv.name();
        if (sdv.getVariableType() == VariableType.CONSTANT || sdv.getVariableType() == VariableType.VARIABLE) {
            return getConstantOrVariable(n);

        }   else {
            VarId inVarId = lookup(n, opInputs, allIterInputs, false);
            Preconditions.checkState(inVarId != null, "Could not find array for variable %s", sdv.name());
            return getTensorFromOutputs(inVarId);
        }
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

        protected OpDep(@NonNull String opName, @NonNull String frame, int iter, FrameIter parentFrame) {
            this.opName = opName;
            this.frame = frame;
            this.iter = iter;
            this.parentFrame = parentFrame;
        }

        @Override
        public String toString() {
            return "OpDep(" + opName + ",frame=" + frame + ",iter=" + iter + (parentFrame == null ? "" : ",parent=" + parentFrame) + ")";
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @AllArgsConstructor
    protected static class PlaceholderDep extends Dep {
        protected String phName;
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @AllArgsConstructor
    protected static class VariableDep extends Dep {
        protected String varName;
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @AllArgsConstructor
    protected static class ConstantDep extends Dep {
        protected String constName;
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @AllArgsConstructor
    protected static class ReqOutputDep extends Dep {
        protected String outputName;
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    @NoArgsConstructor
    protected static class ExecDoneDep extends Dep {
    }
}
