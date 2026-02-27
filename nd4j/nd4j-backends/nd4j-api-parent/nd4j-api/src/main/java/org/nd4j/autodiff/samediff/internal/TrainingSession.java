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

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.ExecutionResult;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.autodiff.samediff.training.LossScaler;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.common.primitives.Pair;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;

@Slf4j
public class TrainingSession extends InferenceSession {

    protected TrainingConfig config;
    protected Map<String, String> gradVarToVarMap;
    protected Map<String, GradientUpdater> updaters;
    protected Map<String, Integer> lossVarsToLossIdx;
    protected double[] currIterLoss;
    protected Map<Class<?>, AtomicDouble> currIterRegLoss;
    protected List<Listener> listeners;

    // Mixed precision training support
    @Getter @Setter
    protected LossScaler lossScaler;
    @Getter
    protected boolean currentIterationOverflow;

    // DSP training support
    private static final boolean DSP_TRAINING_ENABLED = Boolean.parseBoolean(
            System.getProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true"));
    private Map<String, long[]> previousPlaceholderShapes;


    public TrainingSession(SameDiff sameDiff) {
        super(sameDiff);
    }

    /**
     * Perform one iteration of training - i.e., do forward and backward passes, and update the parameters
     *
     * @param config        Training configuration
     * @param placeholders  Current placeholders
     * @param paramsToTrain Set of parameters that will be trained
     * @param updaters      Current updater state
     * @param batch         Current data/batch (mainly for listeners, should have already been converted to placeholders map)
     * @param lossVariables Loss variables (names)
     * @param listeners     Listeners (if any)
     * @param at            Current epoch, iteration, etc
     * @return The Loss at the current iteration
     */
    public Loss trainingIteration(TrainingConfig config, Map<String, INDArray> placeholders, Set<String> paramsToTrain, Map<String, GradientUpdater> updaters,
                                  MultiDataSet batch, List<String> lossVariables, List<Listener> listeners, At at) {
        this.config = config;
        this.updaters = updaters;
        this.currentIterationOverflow = false;

        // Initialize loss scaler if loss scaling is enabled
        if (config.isLossScalingEnabled() && this.lossScaler == null) {
            this.lossScaler = new LossScaler(config.getLossScaleConfig());
        }

        if(batch != null) {
            batch.setCloseable(false);
        }

        //ensure input arrays aren't closed
        if(placeholders != null) {
            placeholders.entrySet().stream().forEach(entry -> {
                entry.getValue().setCloseable(false);
            });
        }

        //Preprocess listeners, get the relevant ones
        if (listeners == null) {
            this.listeners = null;
        } else {
            List<Listener> filtered = new ArrayList<>();
            for (Listener l : listeners) {
                if (l.isActive(at.operation())) {
                    filtered.add(l);
                }
            }
            this.listeners = filtered.isEmpty() ? null : filtered;
        }

        Set<String> requiredActivations = new HashSet<>();
        gradVarToVarMap = new HashMap<>();       //Key: gradient variable. Value: variable that the key is gradient for
        for (String s : paramsToTrain) {
            Preconditions.checkState(sameDiff.hasVariable(s), "SameDiff instance does not have a variable with name \"%s\"", s);
            SDVariable v = sameDiff.getVariable(s);
            Preconditions.checkState(v.getVariableType() == VariableType.VARIABLE, "Can only train VARIABLE type variable - \"%s\" has type %s",
                    s, v.getVariableType());
            SDVariable grad = sameDiff.getVariable(s).getGradient();
            if (grad == null) {
                //In some cases, a variable won't actually impact the loss value, and hence won't have a gradient associated with it
                //For example: floatVar -> cast to integer -> cast to float -> sum -> loss
                //In this case, the gradient of floatVar isn't defined (due to no floating point connection to the loss)
                log.debug("Variable '{}' has no gradient - skipping (no FP connection to loss)", s);
                continue;
            }

            requiredActivations.add(grad.name());

            gradVarToVarMap.put(grad.name(), s);
        }

        //Also add evaluations - in case we want to evaluate something that isn't required to determine loss
        // (hence wouldn't normally be calculated)
        if(config.getTrainEvaluations() != null) {
            requiredActivations.addAll(config.getTrainEvaluations().keySet());
        }

        if(sameDiff.getLossVariables() != null) {
            requiredActivations.addAll(sameDiff.getLossVariables());
        }

        //Set up losses
        lossVarsToLossIdx = new LinkedHashMap<>();
        List<String> lossVars;
        currIterLoss = new double[lossVariables.size()];
        currIterRegLoss = new HashMap<>();
        for (int i = 0; i < lossVariables.size(); i++) {
            lossVarsToLossIdx.put(lossVariables.get(i), i);
        }

        //Do training iteration
        List<String> outputVars = new ArrayList<>(gradVarToVarMap.keySet());    //TODO this should be empty, and grads calculated in requiredActivations
        outputVars.addAll(lossVariables);

        // Try DSP fast path first — bypasses the listener-gated check in output()
        boolean dspHandled = false;
        if (DSP_TRAINING_ENABLED && isDynamicShapePlanEnabled()) {
            try {
                log.debug("Attempting DSP training iteration");
                dspHandled = tryDspTrainingIteration(placeholders, outputVars, requiredActivations, at);
                log.debug("DSP training iteration result: dspHandled={}", dspHandled);
            } catch (Exception e) {
                log.warn("DSP training iteration failed, falling back to standard path: {}", e.getMessage());
                dspHandled = false;
            }
        }

        if (!dspHandled) {
            log.debug("Using standard execution path for training");

            // Execute forward+backward pass via output().
            // When listeners are present (which is always true during training — at minimum
            // ScoreListener/HistoryListener), the per-op execution path is used, which calls
            // getOutputs() on each op. Our getOutputs() override applies updaters inline
            // (gradient → updater → param update) as each gradient op is computed.
            // Therefore we must NOT apply updaters again here.
            //
            // Loss capture also happens inline in getOutputs() — but we also extract from
            // results here as a safety net for the DAG path (listeners empty → DSP).
            Map<String, INDArray> results = output(outputVars, placeholders, batch, requiredActivations, listeners, at);

            // NOTE: Do NOT extract losses or apply updaters here — getOutputs() already
            // handled both inline during per-op execution. Doing it again would:
            //   - Double-count loss values (currIterLoss[i] += l twice)
            //   - Double-update parameters, causing divergence
        }


        double[] finalLoss = new double[currIterLoss.length + currIterRegLoss.size()];
        System.arraycopy(currIterLoss, 0, finalLoss, 0, currIterLoss.length);
        if (currIterRegLoss.size() > 0) {
            lossVars = new ArrayList<>(lossVariables.size() + currIterRegLoss.size());
            lossVars.addAll(lossVariables);
            int s = currIterRegLoss.size();
            //Collect regularization losses
            for (Map.Entry<Class<?>, AtomicDouble> entry : currIterRegLoss.entrySet()) {
                lossVars.add(entry.getKey().getSimpleName());
                finalLoss[s] = entry.getValue().get();
            }
        } else {
            lossVars = lossVariables;
        }

        Loss loss = new Loss(lossVars, finalLoss);
        if (listeners != null) {
            for (Listener l : listeners) {
                if (l.isActive(Operation.TRAINING)) {
                    l.iterationDone(sameDiff, at, batch, loss);
                }
            }
        }

        // Update loss scaler after iteration
        if (lossScaler != null) {
            lossScaler.update(!currentIterationOverflow);
        }

        return loss;
    }

    /**
     * Attempt to execute the training iteration via DynamicShapePlan.
     * This bypasses the listener gate in InferenceSession.output() that prevents DSP
     * from being used during training (since training always has listeners).
     *
     * @return true if DSP execution succeeded; false to fall back to standard path
     */
    private boolean tryDspTrainingIteration(Map<String, INDArray> placeholders,
                                            List<String> outputVars,
                                            Set<String> requiredActivations,
                                            At at) {
        // Build allRequired set (same as output() does)
        Set<String> allRequired = new LinkedHashSet<>(outputVars);
        allRequired.addAll(requiredActivations);

        // Get or build DAG
        ForwardExecutionDAG dag = dagCache.getOrCompute(allRequired, () -> {
            ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sameDiff);
            return builder.buildForwardDAG(allRequired);
        });

        // Enter memory manager scope
        getMmgr().scopeIn();
        try {
            // Lightweight placeholder type casting
            Map<String, INDArray> dspPlaceholders = castPlaceholderTypes(placeholders);

            // Execute via DSP — returns null if DSP is unavailable (not compiled, control flow, etc.)
            Map<String, SDValue> results = executeDynamicShapePlanBased(
                    dag, dspPlaceholders, allRequired, outputVars);
            if (results == null) {
                return false;
            }

            // Extract loss values from results
            log.debug("DSP results keys: {}", results.keySet());
            log.debug("Loss vars to extract: {}", lossVarsToLossIdx.keySet());
            for (Map.Entry<String, Integer> entry : lossVarsToLossIdx.entrySet()) {
                SDValue val = results.get(entry.getKey());
                if (val != null && val.getTensorValue() != null) {
                    INDArray arr = val.getTensorValue();
                    double l = arr.isScalar() ? arr.getDouble(0) : arr.sumNumber().doubleValue();
                    currIterLoss[entry.getValue()] += l;
                    log.debug("DSP loss '{}' = {}", entry.getKey(), l);
                } else {
                    log.warn("DSP loss variable '{}' not found in results (val={})", entry.getKey(), val);
                }
            }

            // Apply updaters post-execution (batch-applied instead of per-op inline)
            applyUpdatersPostDsp(results, at);

            // DSP post-exec: commit + trim
            dspStepCount++;
            DynamicShapePlanExecutor dspExec = dynamicShapePlanExecutorTl.get();
            boolean frozen = dspExec != null && dspExec.isShapesFrozen();

            if (!frozen || dspStepCount <= 2) {
                Nd4j.getExecutioner().commit();
                NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPoolOnStream(
                        Nd4j.getAffinityManager().getDeviceForCurrentThread(), null);
            } else if (dspStepCount % TRIM_INTERVAL == 0) {
                Nd4j.getExecutioner().commit();
                NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPoolOnStream(
                        Nd4j.getAffinityManager().getDeviceForCurrentThread(), null);
            }

            // Manage shape freezing for subsequent iterations
            manageShapeFreezing(placeholders);

            log.debug("DSP training iteration completed successfully");
            return true;
        } finally {
            getMmgr().scopeOut();
        }
    }

    /**
     * Apply updaters and update parameters from standard execution results.
     * The new DAG-based execution engine (executeOperations) bypasses getOutputs(),
     * so updater application must happen post-execution on the result map.
     */
    private void applyUpdatersFromResults(Map<String, INDArray> results, At at) {
        for (Map.Entry<String, String> entry : gradVarToVarMap.entrySet()) {
            String gradName = entry.getKey();
            INDArray gradArr = results.get(gradName);
            if (gradArr == null) {
                continue;
            }
            // Dup the gradient to avoid modifying the execution engine's cached arrays in-place
            applyUpdaterForGradient(entry.getValue(), gradName, gradArr.dup(), at);
        }
    }

    /**
     * Apply updaters and update parameters post-DSP execution.
     * This is equivalent to the updater logic but applied in batch
     * after all ops have been executed by DSP.
     */
    private void applyUpdatersPostDsp(Map<String, SDValue> results, At at) {
        for (Map.Entry<String, String> entry : gradVarToVarMap.entrySet()) {
            String gradName = entry.getKey();
            String varName = entry.getValue();

            SDValue gradValue = results.get(gradName);
            if (gradValue == null || gradValue.getTensorValue() == null) {
                continue;
            }

            applyUpdaterForGradient(varName, gradName, gradValue.getTensorValue(), at);
        }
    }

    /**
     * Core updater logic: apply updater to gradient array and update the parameter.
     * Shared by both DSP and standard execution paths.
     *
     * @param varName  the parameter variable name
     * @param gradName the gradient variable name (for datatype check)
     * @param gradArr  the gradient array
     * @param at       current training position
     */
    private void applyUpdaterForGradient(String varName, String gradName, INDArray gradArr, At at) {
        Variable gradVar = sameDiff.getVariables().get(gradName);
        if (gradVar == null || !gradVar.getVariable().dataType().isFPType())
            return;

        GradientUpdater u = updaters.get(varName);
        Preconditions.checkState(u != null, "No updater found for variable \"%s\"", varName);

        Variable var = sameDiff.getVariables().get(varName);
        INDArray paramArr = var.getVariable().getArr();

        // Unscale gradients if loss scaling is enabled
        if (lossScaler != null) {
            lossScaler.unscaleGradients(gradArr);

            // Check for overflow
            if (!lossScaler.areGradientsFinite(gradArr)) {
                log.debug("Gradient overflow detected for variable: {}", varName);
                currentIterationOverflow = true;
                return;
            }
        }

        // Pre-updater regularization (L1, L2)
        List<Regularization> r = config.getRegularizationForVariable(varName);
        IUpdater varUpdater = config.getUpdaterForVariable(varName);
        if (r != null && r.size() > 0) {
            double lr = varUpdater.hasLearningRate() ? varUpdater.getLearningRate(at.iteration(), at.epoch()) : 1.0;
            for (Regularization reg : r) {
                if (reg.applyStep() == Regularization.ApplyStep.BEFORE_UPDATER) {
                    if (this.listeners != null) {
                        double score = reg.score(paramArr, at.iteration(), at.epoch());
                        if (!currIterRegLoss.containsKey(reg.getClass())) {
                            currIterRegLoss.put(reg.getClass(), new AtomicDouble());
                        }
                        currIterRegLoss.get(reg.getClass()).addAndGet(score);
                    }
                    reg.apply(paramArr, gradArr, lr, at.iteration(), at.epoch());
                }
            }
        }

        u.applyUpdater(gradArr, at.iteration(), at.epoch());

        // Post-apply regularization (weight decay)
        if (r != null && r.size() > 0) {
            double lr = varUpdater.hasLearningRate() ? varUpdater.getLearningRate(at.iteration(), at.epoch()) : 1.0;
            for (Regularization reg : r) {
                if (reg.applyStep() == Regularization.ApplyStep.POST_UPDATER) {
                    if (this.listeners != null) {
                        double score = reg.score(paramArr, at.iteration(), at.epoch());
                        if (!currIterRegLoss.containsKey(reg.getClass())) {
                            currIterRegLoss.put(reg.getClass(), new AtomicDouble());
                        }
                        currIterRegLoss.get(reg.getClass()).addAndGet(score);
                    }
                    reg.apply(paramArr, gradArr, lr, at.iteration(), at.epoch());
                }
            }
        }

        if (this.listeners != null) {
            for (Listener l : this.listeners) {
                if (l.isActive(at.operation()))
                    l.preUpdate(sameDiff, at, var, gradArr);
            }
        }

        // Update parameter
        if (config.isMinimize()) {
            paramArr.subi(gradArr);
        } else {
            paramArr.addi(gradArr);
        }

        log.trace("Applied updater to gradient and updated variable: {}", varName);
    }

    /**
     * Manage shape freezing for training iterations.
     * Training batches within an epoch have consistent shapes, so we freeze eagerly
     * after the first successful DSP execution to enable CUDA graph replay from
     * iteration 1 onward. If a shape change occurs (incomplete last batch, new epoch
     * with different data), the executor detects the mismatch and we unfreeze here.
     */
    private void manageShapeFreezing(Map<String, INDArray> placeholders) {
        if (placeholders == null || placeholders.isEmpty()) {
            return;
        }

        DynamicShapePlanExecutor executor = dynamicShapePlanExecutorTl.get();
        if (executor == null) {
            return;
        }

        // Build current shape map
        Map<String, long[]> currentShapes = new HashMap<>();
        for (Map.Entry<String, INDArray> e : placeholders.entrySet()) {
            if (e.getValue() != null) {
                currentShapes.put(e.getKey(), e.getValue().shape());
            }
        }

        // Eager freeze: training batches are same-shaped within an epoch.
        // Freeze after first execution so iteration 1+ gets graph replay.
        // If shapes change (last incomplete batch, new epoch), unfreeze.
        if (previousPlaceholderShapes == null) {
            // First iteration: record shapes and freeze immediately
            if (!executor.isShapesFrozen()) {
                executor.setShapesFrozen(true);
                log.info("DSP training: froze shapes eagerly after first iteration");
            }
        } else if (!shapesMatch(previousPlaceholderShapes, currentShapes)) {
            // Shape changed: unfreeze so executor recomputes shapes
            if (executor.isShapesFrozen()) {
                executor.setShapesFrozen(false);
                log.info("DSP training: unfroze shapes due to shape change");
            }
            // Re-freeze immediately for the new shape — next iteration will
            // likely have the same shape again (new epoch, consistent batches)
            executor.setShapesFrozen(true);
            log.info("DSP training: re-froze shapes for new batch shape");
        }
        // If shapes match previous and already frozen: nothing to do (graph replay)

        previousPlaceholderShapes = currentShapes;
    }

    private static boolean shapesMatch(Map<String, long[]> a, Map<String, long[]> b) {
        if (a.size() != b.size()) return false;
        for (Map.Entry<String, long[]> e : a.entrySet()) {
            long[] bShape = b.get(e.getKey());
            if (bShape == null || !Arrays.equals(e.getValue(), bShape)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public ExecutionResult getOutputs(Pair<SameDiffOp, OpContext> opPair, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                      Set<String> constAndPhInputs, List<Listener> listeners, At at, MultiDataSet batch, Set<String> allReqVariables, Map<String, SDValue> otherPlaceHolders) {
        //Get outputs from InferenceSession
        ExecutionResult out = super.getOutputs(opPair, outputFrameIter, opInputs, allIterInputs, constAndPhInputs, listeners, at, batch, allReqVariables, otherPlaceHolders);
        SameDiffOp op = opPair.getFirst();

        List<String> outputs = op.getOutputsOfOp();
        log.debug("getOutputs for op '{}': outputs={}", op.getName(), outputs);
        int outIdx = 0;
        for (String s : outputs) {
            //If this is a loss variable - record it
            if (lossVarsToLossIdx.containsKey(s)) {
                int lossIdx = lossVarsToLossIdx.get(s);
                INDArray arr = out.resultAt(outIdx);
                double l = arr.isScalar() ? arr.getDouble(0) : arr.sumNumber().doubleValue();
                currIterLoss[lossIdx] += l;
                log.debug("Captured loss '{}' = {} (lossIdx={})", s, l, lossIdx);
            }

            //If this is a gradient variable - apply the updater and update the parameter array in-line
            if (gradVarToVarMap.containsKey(s)) {
                String varName = gradVarToVarMap.get(s);
                //log.info("Calculated gradient for variable \"{}\": (grad var name: \"{}\")", varName, s);

                Variable gradVar = sameDiff.getVariables().get(s);
                if(!gradVar.getVariable().dataType().isFPType())
                    continue;
                if (gradVar.getInputsForOp() != null && gradVar.getInputsForOp().isEmpty()) {
                    //Should be rare, and we should handle this by tracking dependencies, and only update when safe
                    // (i.e., dependency tracking)
                    throw new IllegalStateException("Op depends on gradient variable: " + s + " for variable " + varName);
                }

                GradientUpdater u = updaters.get(varName);
                Preconditions.checkState(u != null, "No updater found for variable \"%s\"", varName);

                Variable var = sameDiff.getVariables().get(varName);
                INDArray gradArr = out.resultAt(outIdx);
                INDArray paramArr = var.getVariable().getArr();

                // Unscale gradients if loss scaling is enabled
                if (lossScaler != null) {
                    lossScaler.unscaleGradients(gradArr);

                    // Check for overflow
                    if (!lossScaler.areGradientsFinite(gradArr)) {
                        log.debug("Gradient overflow detected for variable: {}", varName);
                        currentIterationOverflow = true;
                        outIdx++;
                        continue; // Skip this parameter update
                    }
                }

                //Pre-updater regularization (L1, L2)
                // Use per-variable regularization if available
                List<Regularization> r = config.getRegularizationForVariable(varName);
                // Get learning rate from per-variable updater if available
                IUpdater varUpdater = config.getUpdaterForVariable(varName);
                if (r != null && r.size() > 0) {
                    double lr = varUpdater.hasLearningRate() ? varUpdater.getLearningRate(at.iteration(), at.epoch()) : 1.0;
                    for (Regularization reg : r) {
                        if (reg.applyStep() == Regularization.ApplyStep.BEFORE_UPDATER) {
                            if (this.listeners != null) {
                                double score = reg.score(paramArr, at.iteration(), at.epoch());
                                if (!currIterRegLoss.containsKey(reg.getClass())) {
                                    currIterRegLoss.put(reg.getClass(), new AtomicDouble());
                                }
                                currIterRegLoss.get(reg.getClass()).addAndGet(score);
                            }
                            reg.apply(paramArr, gradArr, lr, at.iteration(), at.epoch());
                        }
                    }
                }

                u.applyUpdater(gradArr, at.iteration(), at.epoch());

                //Post-apply regularization (weight decay)
                if (r != null && r.size() > 0) {
                    double lr = varUpdater.hasLearningRate() ? varUpdater.getLearningRate(at.iteration(), at.epoch()) : 1.0;
                    for (Regularization reg : r) {
                        if (reg.applyStep() == Regularization.ApplyStep.POST_UPDATER) {
                            if (this.listeners != null) {
                                double score = reg.score(paramArr, at.iteration(), at.epoch());
                                if (!currIterRegLoss.containsKey(reg.getClass())) {
                                    currIterRegLoss.put(reg.getClass(), new AtomicDouble());
                                }
                                currIterRegLoss.get(reg.getClass()).addAndGet(score);
                            }
                            reg.apply(paramArr, gradArr, lr, at.iteration(), at.epoch());
                        }
                    }
                }

                if (listeners != null) {
                    for (Listener l : listeners) {
                        if (l.isActive(at.operation()))
                            l.preUpdate(sameDiff, at, var, gradArr);
                    }
                }

                //Update:
                if (config.isMinimize()) {
                    paramArr.subi(gradArr);
                } else {
                    paramArr.addi(gradArr);
                }

                log.trace("Applied updater to gradient and updated variable: {}", varName);
            }

            outIdx++;
        }

        return out;
    }

    /**
     * Check if the current iteration had a gradient overflow.
     * This is useful for mixed precision training to detect when gradients
     * have become too large (inf/nan) due to loss scaling.
     *
     * @return true if overflow was detected in this iteration
     */
    public boolean hadOverflow() {
        return currentIterationOverflow;
    }
}
