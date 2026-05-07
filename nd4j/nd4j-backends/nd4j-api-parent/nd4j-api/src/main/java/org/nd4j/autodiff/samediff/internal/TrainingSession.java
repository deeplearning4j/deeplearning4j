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
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.autodiff.samediff.execution.ReplayProfileManager;
import org.nd4j.autodiff.samediff.execution.UpdaterOpsAppender;
import org.nd4j.autodiff.samediff.training.LossScaler;
import org.nd4j.common.base.Preconditions;
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

    // DSP training support — always enabled when DSP is available
    private Map<String, long[]> previousPlaceholderShapes;

    // Replay profile collection (persists across shape changes within training run)
    private ReplayProfileManager.ReplayProfileCollection replayProfiles_;
    private int dspStepCount;

    // Gradient accumulation state
    // Key: gradient variable name. Value: accumulated (summed) gradient INDArray.
    private Map<String, INDArray> accumulatedGradients;
    // How many micro-batches have been accumulated so far in the current cycle.
    private int accumulationStep;


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
        if (isDynamicShapePlanEnabled()) {
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

            // Execute forward+backward pass via output(). The new DAG-based execution engine
            // (executeOperations → executeNode) bypasses TrainingSession.getOutputs() entirely,
            // so loss capture and updater application must happen here post-execution.
            Map<String, INDArray> results = output(outputVars, placeholders, batch, requiredActivations, listeners, at);

            // Extract loss values from results
            for (Map.Entry<String, Integer> entry : lossVarsToLossIdx.entrySet()) {
                INDArray arr = results.get(entry.getKey());
                if (arr != null) {
                    double l = arr.isScalar() ? arr.getDouble(0) : arr.sumNumber().doubleValue();
                    currIterLoss[entry.getValue()] += l;
                    log.debug("Captured loss '{}' = {} (lossIdx={})", entry.getKey(), l, entry.getValue());
                }
            }

            // Apply updaters and update parameters post-execution
            applyUpdatersFromResults(results, at);
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
     * <p>When updater fusion is active, the grad DSP plan already contains the updater
     * ops and weight-assign ops. In that case this method adds the weight-updated output
     * names to the requested outputs, and after execution writes the updated weight
     * tensors back to the SDVariable arrays instead of calling the Java-side updater.</p>
     *
     * @return true if DSP execution succeeded; false to fall back to standard path
     */
    private boolean tryDspTrainingIteration(Map<String, INDArray> placeholders,
                                            List<String> outputVars,
                                            Set<String> requiredActivations,
                                            At at) {
        // Check if updater ops are fused into the grad graph.
        UpdaterOpsAppender.AppendResult fusionResult = sameDiff.getUpdaterFusionResult();
        boolean fusionActive = fusionResult != null && !fusionResult.varToWeightUpdatedOutput.isEmpty();

        // When fusion is active, request the weight-updated output variables AND the new
        // state-value outputs from the plan, so they appear in the results map after execution.
        List<String> effectiveOutputVars = outputVars;
        if (fusionActive) {
            effectiveOutputVars = new ArrayList<>(outputVars);
            effectiveOutputVars.addAll(fusionResult.varToWeightUpdatedOutput.values());
            // State-value outputs must also be requested so the plan surfaces them.
            if (fusionResult.stateVarToNewOutput != null) {
                effectiveOutputVars.addAll(fusionResult.stateVarToNewOutput.values());
            }
        }

        // Build allRequired set (same as output() does)
        Set<String> allRequired = new LinkedHashSet<>(effectiveOutputVars);
        allRequired.addAll(requiredActivations);

        // Get or build DAG
        final List<String> finalEffectiveOutputVars = effectiveOutputVars;
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
                    dag, dspPlaceholders, allRequired, finalEffectiveOutputVars);
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

            if (fusionActive) {
                // Updater ops ran inside the plan — apply updated weights back to SDVariables.
                applyFusedWeightUpdates(fusionResult, results, at);
            } else {
                // Apply updaters post-execution (batch-applied instead of per-op inline)
                applyUpdatersPostDsp(results, at);
            }

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
     * Collect gradient arrays from standard execution results into a map.
     * Returns a map from gradient variable name to a .dup() copy of the gradient array.
     * Variables whose gradients are missing from results or are not FP type are skipped.
     * If loss scaling is active, gradients are unscaled here (before accumulation).
     * Sets {@code currentIterationOverflow} to true if any gradient overflows.
     */
    private Map<String, INDArray> collectGradientsFromResults(Map<String, INDArray> results, At at) {
        Map<String, INDArray> gradients = new LinkedHashMap<>();
        int applied = 0;
        int skipped = 0;
        for (Map.Entry<String, String> entry : gradVarToVarMap.entrySet()) {
            String gradName = entry.getKey();
            String varName = entry.getValue();
            INDArray gradArr = results.get(gradName);
            if (gradArr == null) {
                log.warn("Gradient '{}' not found in results (available keys: {})", gradName,
                    results.keySet().stream().limit(20).collect(java.util.stream.Collectors.toList()));
                skipped++;
                continue;
            }
            Variable gradVar = sameDiff.getVariables().get(gradName);
            if (gradVar == null || !gradVar.getVariable().dataType().isFPType()) {
                skipped++;
                continue;
            }
            if (at.iteration() < 5) {
                log.trace("Raw gradient '{}' for var '{}': shape={}, mean={}, absMax={}, min={}, max={}",
                    gradName, varName, java.util.Arrays.toString(gradArr.shape()),
                    gradArr.meanNumber(), gradArr.ameanNumber(),
                    gradArr.minNumber(), gradArr.maxNumber());
            }
            // Dup the gradient to avoid modifying the execution engine's cached arrays in-place
            INDArray gradCopy = gradArr.dup();
            // Unscale here — before accumulation — so accumulated values are in true gradient space
            if (lossScaler != null) {
                lossScaler.unscaleGradients(gradCopy);
                if (!lossScaler.areGradientsFinite(gradCopy)) {
                    log.debug("Gradient overflow detected for variable '{}' during collection", varName);
                    currentIterationOverflow = true;
                    skipped++;
                    continue;
                }
            }
            gradients.put(gradName, gradCopy);
            applied++;
        }
        if (at.iteration() < 5) {
            log.trace("collectGradientsFromResults: collected={}, skipped={}, gradVarToVarMap.size={}", applied, skipped, gradVarToVarMap.size());
        }
        return gradients;
    }

    /**
     * Collect gradient arrays from DSP execution results into a map.
     * Returns a map from gradient variable name to a .dup() copy of the gradient array.
     * Variables whose gradients are missing from results or are not FP type are skipped.
     * If loss scaling is active, gradients are unscaled here (before accumulation).
     * Sets {@code currentIterationOverflow} to true if any gradient overflows.
     */
    private Map<String, INDArray> collectGradientsFromDsp(Map<String, SDValue> results, At at) {
        Map<String, INDArray> gradients = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : gradVarToVarMap.entrySet()) {
            String gradName = entry.getKey();
            String varName = entry.getValue();
            SDValue gradValue = results.get(gradName);
            if (gradValue == null || gradValue.getTensorValue() == null) {
                continue;
            }
            Variable gradVar = sameDiff.getVariables().get(gradName);
            if (gradVar == null || !gradVar.getVariable().dataType().isFPType()) {
                continue;
            }
            // Dup to own the array independently of DSP result buffers
            INDArray gradCopy = gradValue.getTensorValue().dup();
            // Unscale here — before accumulation — so accumulated values are in true gradient space
            if (lossScaler != null) {
                lossScaler.unscaleGradients(gradCopy);
                if (!lossScaler.areGradientsFinite(gradCopy)) {
                    log.debug("Gradient overflow detected for variable '{}' during DSP collection", varName);
                    currentIterationOverflow = true;
                    continue;
                }
            }
            gradients.put(gradName, gradCopy);
        }
        return gradients;
    }

    /**
     * Accumulate the given collected gradients into {@code accumulatedGradients}.
     * On the first micro-batch of a new cycle the map is created fresh.
     * If an incoming gradient has a different shape than the already-accumulated gradient
     * (which should not happen in normal training but is guarded here defensively),
     * the accumulated state is cleared and a warning is logged.
     *
     * @param collected gradients collected from the current micro-batch execution
     */
    private void accumulateGradients(Map<String, INDArray> collected) {
        if (accumulatedGradients == null) {
            // First micro-batch: take ownership of the collected arrays directly.
            // Callers (collectGradientsFromResults / collectGradientsFromDsp) already dup'd them,
            // so no second dup is needed here — we just own the map.
            accumulatedGradients = new LinkedHashMap<>(collected);
            return;
        }
        // Subsequent micro-batches: add in-place
        for (Map.Entry<String, INDArray> e : collected.entrySet()) {
            INDArray existing = accumulatedGradients.get(e.getKey());
            if (existing == null) {
                // New gradient variable appeared mid-cycle (shouldn't happen, be safe)
                accumulatedGradients.put(e.getKey(), e.getValue().dup());
            } else if (!java.util.Arrays.equals(existing.shape(), e.getValue().shape())) {
                log.warn("Gradient shape changed mid-accumulation for '{}': expected {} got {}. " +
                         "Clearing accumulated state and restarting accumulation cycle.",
                        e.getKey(), java.util.Arrays.toString(existing.shape()),
                        java.util.Arrays.toString(e.getValue().shape()));
                accumulatedGradients = null;
                accumulationStep = 0;
                // Restart with just this micro-batch (collected values are already dup'd)
                accumulateGradients(collected);
                return;
            } else {
                existing.addi(e.getValue());
            }
        }
    }

    /**
     * Apply the accumulated gradients: divide by N, then apply updater and weight update
     * for each gradient variable, then clear the accumulated state.
     *
     * Regularization and updater logic are delegated to {@link #applyUpdaterForGradientPreUnscaled},
     * which expects gradients that have ALREADY been unscaled (done during collection).
     *
     * @param n   number of micro-batches that were accumulated (divisor)
     * @param at  current training position
     */
    private void applyAccumulatedGradients(int n, At at) {
        if (accumulatedGradients == null || accumulatedGradients.isEmpty()) {
            return;
        }
        for (Map.Entry<String, INDArray> e : accumulatedGradients.entrySet()) {
            String gradName = e.getKey();
            String varName = gradVarToVarMap.get(gradName);
            if (varName == null) {
                continue;
            }
            INDArray accGrad = e.getValue();
            // Divide by number of accumulated micro-batches to get the mean gradient
            if (n > 1) {
                accGrad.divi(n);
            }
            // Delegate to the pre-unscaled updater path (loss scaling already applied at collect time)
            applyUpdaterForGradientPreUnscaled(varName, gradName, accGrad, at);
        }
        accumulatedGradients = null;
        accumulationStep = 0;
        log.debug("Applied accumulated gradients over {} micro-batches", n);
    }

    /**
     * Apply updaters and update parameters from standard execution results.
     * The new DAG-based execution engine (executeOperations) bypasses getOutputs(),
     * so updater application must happen post-execution on the result map.
     */
    private void applyUpdatersFromResults(Map<String, INDArray> results, At at) {
        int n = config.getGradientAccumulationSteps();
        if (n <= 1) {
            // Fast path: no accumulation — collect and apply immediately.
            // If the legacy getOutputs() path already pre-populated accumulatedGradients
            // (which should not happen when n<=1, but guard defensively), ignore them.
            Map<String, INDArray> collected = collectGradientsFromResults(results, at);
            for (Map.Entry<String, INDArray> e : collected.entrySet()) {
                String gradName = e.getKey();
                String varName = gradVarToVarMap.get(gradName);
                if (varName != null) {
                    // Loss scaler already handled in collectGradientsFromResults; use pre-unscaled path
                    applyUpdaterForGradientPreUnscaled(varName, gradName, e.getValue(), at);
                }
            }
        } else {
            // Gradient accumulation path.
            //
            // The legacy getOutputs() path (AbstractSession op-by-op loop) accumulates gradients
            // directly into accumulatedGradients as it encounters each gradient op.  In that case
            // accumulatedGradients is non-null here and already contains the correctly unscaled
            // gradients for this micro-batch — we must NOT call collectGradientsFromResults again
            // (which would double-add from the results map).
            //
            // The new DAG-based executeOperations path does NOT call getOutputs() at all, so
            // accumulatedGradients is null here and we collect from the results map as normal.
            boolean prePopulatedByGetOutputs = (accumulatedGradients != null);

            if (prePopulatedByGetOutputs) {
                // Legacy path: gradients were already accumulated by getOutputs().
                // Just drive the step counter and apply if the cycle is complete.
                if (!currentIterationOverflow) {
                    accumulationStep++;
                    log.debug("Gradient accumulation (legacy path): step {}/{}", accumulationStep, n);
                    if (accumulationStep >= n) {
                        applyAccumulatedGradients(n, at);
                    }
                } else {
                    log.debug("Gradient overflow (legacy path) during accumulation step {}; discarding micro-batch", accumulationStep);
                    accumulatedGradients = null;
                    accumulationStep = 0;
                }
            } else {
                // New DAG path: collect gradients from the results map.
                Map<String, INDArray> collected = collectGradientsFromResults(results, at);
                if (!currentIterationOverflow) {
                    accumulateGradients(collected);
                    accumulationStep++;
                    log.debug("Gradient accumulation: step {}/{}", accumulationStep, n);
                    if (accumulationStep >= n) {
                        applyAccumulatedGradients(n, at);
                    }
                } else {
                    // Overflow: discard this micro-batch, reset accumulation cycle
                    log.debug("Gradient overflow in standard path during accumulation step {}; discarding micro-batch", accumulationStep);
                    accumulatedGradients = null;
                    accumulationStep = 0;
                }
            }
        }
    }

    /**
     * Apply updaters and update parameters post-DSP execution.
     * This is equivalent to the updater logic but applied in batch
     * after all ops have been executed by DSP.
     */
    private void applyUpdatersPostDsp(Map<String, SDValue> results, At at) {
        int n = config.getGradientAccumulationSteps();
        if (n <= 1) {
            // Fast path: no accumulation — collect and apply immediately
            Map<String, INDArray> collected = collectGradientsFromDsp(results, at);
            for (Map.Entry<String, INDArray> e : collected.entrySet()) {
                String gradName = e.getKey();
                String varName = gradVarToVarMap.get(gradName);
                if (varName != null) {
                    applyUpdaterForGradientPreUnscaled(varName, gradName, e.getValue(), at);
                }
            }
        } else {
            // Gradient accumulation path
            Map<String, INDArray> collected = collectGradientsFromDsp(results, at);
            if (!currentIterationOverflow) {
                accumulateGradients(collected);
                accumulationStep++;
                log.debug("DSP gradient accumulation: step {}/{}", accumulationStep, n);
                if (accumulationStep >= n) {
                    applyAccumulatedGradients(n, at);
                }
            } else {
                // Overflow: discard this micro-batch, reset accumulation cycle
                log.debug("Gradient overflow in DSP path during accumulation step {}; discarding micro-batch", accumulationStep);
                accumulatedGradients = null;
                accumulationStep = 0;
            }
        }
    }

    /**
     * Apply fused weight updates after DSP execution when updater ops are included in the plan.
     *
     * <p>When updater fusion is active, the DSP plan executed the updater ops and the
     * weight-delta ops (sub/add) in C++. This method:
     * <ol>
     *   <li>Reads the new-weight output (weight ± scaledGrad) from the plan results and
     *       copies it into the weight SDVariable's backing array.</li>
     *   <li>Reads the new-state outputs (new moment estimates, etc.) and copies them back
     *       into the state SDVariable backing arrays so the next iteration uses updated state.</li>
     * </ol>
     * </p>
     *
     * <p>For variables that were skipped by the fusion (unsupported updater type etc.) we
     * fall back to the Java-side updater path for those variables only.</p>
     *
     * @param fusionResult  Result from {@link UpdaterOpsAppender#appendUpdaterOps}.
     * @param results       DSP execution results map (variable name → SDValue).
     * @param at            Current training position.
     */
    private void applyFusedWeightUpdates(UpdaterOpsAppender.AppendResult fusionResult,
                                         Map<String, SDValue> results,
                                         At at) {
        int applied = 0;
        int missing = 0;

        for (Map.Entry<String, String> e : fusionResult.varToWeightUpdatedOutput.entrySet()) {
            String varName = e.getKey();
            String updatedOutputName = e.getValue();

            SDValue updatedVal = results.get(updatedOutputName);
            if (updatedVal == null || updatedVal.getTensorValue() == null) {
                log.warn("DSP updater fusion: updated weight output '{}' for variable '{}' " +
                         "not found in results, falling back to Java-side updater for this variable",
                        updatedOutputName, varName);
                missing++;
                // Fall back to Java-side updater for this variable.
                String gradName = varName + "-grad";
                SDValue gradVal = results.get(gradName);
                if (gradVal != null && gradVal.getTensorValue() != null) {
                    INDArray gradArr = gradVal.getTensorValue().dup();
                    if (lossScaler != null) {
                        lossScaler.unscaleGradients(gradArr);
                        if (!lossScaler.areGradientsFinite(gradArr)) {
                            currentIterationOverflow = true;
                            continue;
                        }
                    }
                    applyUpdaterForGradientPreUnscaled(varName, gradName, gradArr, at);
                }
                continue;
            }

            // Write the updated weight tensor back to the SDVariable's array.
            Variable var = sameDiff.getVariables().get(varName);
            if (var == null || var.getVariable() == null) {
                log.warn("DSP updater fusion: weight variable '{}' not found in grad graph, skipping", varName);
                missing++;
                continue;
            }

            INDArray paramArr = var.getVariable().getArr();
            if (paramArr == null) {
                log.warn("DSP updater fusion: weight variable '{}' has null array, skipping", varName);
                missing++;
                continue;
            }

            // Copy the new weight value from the plan output into the parameter's backing array.
            // The plan produced newWeight = weight ± updatedGrad as a new output variable;
            // we write it back here so the next iteration sees the updated weights.
            INDArray updatedWeight = updatedVal.getTensorValue();
            if (paramArr != updatedWeight && !paramArr.data().equals(updatedWeight.data())) {
                paramArr.assign(updatedWeight);
            }

            applied++;
            log.trace("DSP updater fusion: updated weight '{}' from plan output '{}'",
                    varName, updatedOutputName);
        }

        // Sync updater state arrays (moment estimates, running averages, etc.) back to their
        // SDVariable backing arrays so the next execution sees the updated state.
        int stateApplied = 0;
        int stateMissing = 0;
        if (fusionResult.stateVarToNewOutput != null) {
            for (Map.Entry<String, String> se : fusionResult.stateVarToNewOutput.entrySet()) {
                String stateVarName = se.getKey();
                String newStateOutputName = se.getValue();

                SDValue newStateVal = results.get(newStateOutputName);
                if (newStateVal == null || newStateVal.getTensorValue() == null) {
                    log.debug("DSP updater fusion: new state output '{}' for state '{}' " +
                              "not found in results (may be pruned by plan)", newStateOutputName, stateVarName);
                    stateMissing++;
                    continue;
                }

                Variable stateVar = sameDiff.getVariables().get(stateVarName);
                if (stateVar == null || stateVar.getVariable() == null) {
                    log.debug("DSP updater fusion: state variable '{}' not found in grad graph", stateVarName);
                    stateMissing++;
                    continue;
                }

                INDArray stateArr = stateVar.getVariable().getArr();
                if (stateArr == null) {
                    stateMissing++;
                    continue;
                }

                INDArray newState = newStateVal.getTensorValue();
                if (stateArr != newState && !stateArr.data().equals(newState.data())) {
                    stateArr.assign(newState);
                }
                stateApplied++;
            }
        }
        log.debug("DSP updater fusion state sync: applied={}, missing={}", stateApplied, stateMissing);

        // For variables skipped during fusion (unsupported updater type), apply Java-side.
        if (!fusionResult.skippedVars.isEmpty()) {
            for (String skippedVar : fusionResult.skippedVars) {
                String gradName = skippedVar + "-grad";
                SDValue gradVal = results.get(gradName);
                if (gradVal != null && gradVal.getTensorValue() != null) {
                    INDArray gradArr = gradVal.getTensorValue().dup();
                    if (lossScaler != null) {
                        lossScaler.unscaleGradients(gradArr);
                        if (!lossScaler.areGradientsFinite(gradArr)) {
                            currentIterationOverflow = true;
                            continue;
                        }
                    }
                    applyUpdaterForGradientPreUnscaled(skippedVar, gradName, gradArr, at);
                }
            }
        }

        log.debug("DSP updater fusion: applied={}, missing={}, skippedFallback={}",
                applied, missing, fusionResult.skippedVars.size());
    }

    /**
     * Core updater logic (with loss-scale unscaling): apply updater to gradient array
     * and update the parameter. Called from the legacy inline {@code getOutputs()} path,
     * which has not yet unscaled the gradient.
     *
     * For the standard and DSP paths, use {@link #applyUpdaterForGradientPreUnscaled}
     * instead, since those paths now unscale during gradient collection.
     *
     * @param varName  the parameter variable name
     * @param gradName the gradient variable name (for datatype check)
     * @param gradArr  the gradient array (not yet unscaled)
     * @param at       current training position
     */
    private void applyUpdaterForGradient(String varName, String gradName, INDArray gradArr, At at) {
        Variable gradVar = sameDiff.getVariables().get(gradName);
        if (gradVar == null || !gradVar.getVariable().dataType().isFPType())
            return;

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

        applyUpdaterForGradientPreUnscaled(varName, gradName, gradArr, at);
    }

    /**
     * Core updater logic (pre-unscaled): apply updater to gradient array and update the
     * parameter. The gradient array must already have been unscaled (or loss scaling is
     * not in use). This is the version called from the accumulation path and from the
     * post-collection fast paths, since unscaling is performed during gradient collection.
     *
     * @param varName  the parameter variable name
     * @param gradName the gradient variable name (for datatype check)
     * @param gradArr  the gradient array (already unscaled if applicable)
     * @param at       current training position
     */
    private void applyUpdaterForGradientPreUnscaled(String varName, String gradName, INDArray gradArr, At at) {
        Variable gradVar = sameDiff.getVariables().get(gradName);
        if (gradVar == null || !gradVar.getVariable().dataType().isFPType())
            return;

        GradientUpdater u = updaters.get(varName);
        Preconditions.checkState(u != null, "No updater found for variable \"%s\"", varName);

        Variable var = sameDiff.getVariables().get(varName);
        INDArray paramArr = var.getVariable().getArr();

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

        if (at.iteration() == 0) {
            log.trace("Pre-updater grad '{}': shape={}, values={}", varName,
                java.util.Arrays.toString(gradArr.shape()), gradArr.toStringFull());
        }

        u.applyUpdater(gradArr, at.iteration(), at.epoch());

        if (at.iteration() == 0) {
            log.trace("Post-updater grad '{}': values={}", varName, gradArr.toStringFull());
        }

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
     * with different data), the current native plan is destroyed and a fresh one is
     * compiled for the new shape on the next iteration.
     * <p>
     * Plan phases are strictly linear: SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING.
     * Backward transitions (unfreezing) are illegal. When shapes change, the only correct
     * action is to destroy the existing plan and let the cache produce a fresh entry for
     * the new shape.
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
        // Freeze after first successful execution so iteration 1+ gets graph replay.
        // If shapes change (last incomplete batch, new epoch with different data),
        // destroy the current plan — phases are linear and immutable, so unfreezing
        // is illegal. The plan cache will compile a fresh entry for the new shape.
        if (previousPlaceholderShapes == null) {
            // First iteration: freeze immediately so subsequent same-shaped iterations
            // get graph replay.
            if (!executor.isShapesFrozen()) {
                executor.setShapesFrozen(true);
                log.info("DSP training: froze shapes eagerly after first iteration");
            }
        } else if (!shapesMatch(previousPlaceholderShapes, currentShapes)) {
            // Shape changed. Destroy the current plan (releasing GPU intermediates and
            // unpinning the handle back to the cache). Reset previousPlaceholderShapes
            // to null so the next call treats it as a "first iteration" and freezes after
            // the new plan executes successfully.
            log.info("DSP training: shape change detected — destroying plan for new shape");
            executor.resetForNextPage();
            previousPlaceholderShapes = null;
            dspStepCount++;
            return;
        }
        // If shapes match previous and already frozen: nothing to do (graph replay)

        previousPlaceholderShapes = currentShapes;
        dspStepCount++;

        // Replay profile management for warm restarts
        if (replayProfiles_ != null && executor.isShapesFrozen()) {
            org.bytedeco.javacpp.Pointer planHandle = executor.getNativePlanHandle();
            if (planHandle != null && dspStepCount > 3) {
                // Wait for capture to stabilize before snapshotting profile
                ReplayProfileManager.ReplayProfile profile =
                    ReplayProfileManager.captureProfile(planHandle, currentShapes);
                replayProfiles_.put(profile);
            }
        }
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

    /**
     * Enable replay profile tracking for warm restarts.
     */
    public void setReplayProfilesEnabled(boolean enabled) {
        if (enabled && replayProfiles_ == null) {
            replayProfiles_ = new ReplayProfileManager.ReplayProfileCollection();
        } else if (!enabled) {
            replayProfiles_ = null;
        }
    }

    /**
     * Get replay profiles (for checkpoint saving).
     */
    public ReplayProfileManager.ReplayProfileCollection getReplayProfiles() {
        return replayProfiles_;
    }

    /**
     * Load replay profiles (for warm restart from checkpoint).
     */
    public void loadReplayProfiles(ReplayProfileManager.ReplayProfileCollection profiles) {
        this.replayProfiles_ = profiles;
        DynamicShapePlanExecutor executor = dynamicShapePlanExecutorTl.get();
        if (executor != null && profiles != null) {
            org.bytedeco.javacpp.Pointer planHandle = executor.getNativePlanHandle();
            if (planHandle != null && previousPlaceholderShapes != null) {
                ReplayProfileManager.ReplayProfile match =
                    ReplayProfileManager.findMatchingProfile(profiles, previousPlaceholderShapes);
                if (match != null) {
                    ReplayProfileManager.loadProfile(planHandle, match);
                    log.info("DSP training: loaded replay profile for shape hash {}",
                             match.getShapeHash());
                }
            }
        }
    }

    /**
     * Save replay profiles to disk alongside model checkpoint.
     */
    public void saveReplayProfilesToCheckpoint(String checkpointDir) {
        if (replayProfiles_ != null) {
            replayProfiles_.saveToDisk(checkpointDir);
        }
    }

    /**
     * Load replay profiles from checkpoint directory.
     */
    public void loadReplayProfilesFromCheckpoint(String checkpointDir) {
        ReplayProfileManager.ReplayProfileCollection loaded =
            ReplayProfileManager.ReplayProfileCollection.loadFromDisk(checkpointDir);
        if (loaded != null) {
            loadReplayProfiles(loaded);
        }
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

            //If this is a gradient variable — handle updater application or gradient accumulation.
            // NOTE: The new DAG-based execution engine (executeOperations) bypasses getOutputs()
            // entirely; in that case, applyUpdatersFromResults() handles everything post-execution.
            // getOutputs() is only entered for the legacy InferenceSession op-by-op path.
            if (gradVarToVarMap.containsKey(s)) {
                String varName = gradVarToVarMap.get(s);

                Variable gradVar = sameDiff.getVariables().get(s);
                if(!gradVar.getVariable().dataType().isFPType())
                    continue;
                if (gradVar.getInputsForOp() != null && gradVar.getInputsForOp().isEmpty()) {
                    //Should be rare, and we should handle this by tracking dependencies, and only update when safe
                    // (i.e., dependency tracking)
                    throw new IllegalStateException("Op depends on gradient variable: " + s + " for variable " + varName);
                }

                INDArray gradArr = out.resultAt(outIdx).dup(); // dup to own the array

                int n = config.getGradientAccumulationSteps();
                if (n <= 1) {
                    // No accumulation: apply updater immediately (original behaviour).
                    // applyUpdaterForGradient handles unscaling internally.
                    applyUpdaterForGradient(varName, s, gradArr, at);
                } else {
                    // Gradient accumulation: collect into accumulatedGradients.
                    // The map is keyed by gradient variable name so applyUpdatersFromResults
                    // (which runs after output() returns) can look up each gradient by name.
                    // Unscale before accumulation so accumulated values are in true gradient space.
                    if (lossScaler != null) {
                        lossScaler.unscaleGradients(gradArr);
                        if (!lossScaler.areGradientsFinite(gradArr)) {
                            log.debug("Gradient overflow detected for variable '{}' in getOutputs (accumulation step {})",
                                    varName, accumulationStep);
                            currentIterationOverflow = true;
                            outIdx++;
                            continue;
                        }
                    }
                    // Accumulate; accumulationStep is incremented once per micro-batch in
                    // applyUpdatersFromResults after getOutputs returns, so we just collect here.
                    if (accumulatedGradients == null) {
                        accumulatedGradients = new LinkedHashMap<>();
                    }
                    INDArray existing = accumulatedGradients.get(s);
                    if (existing == null) {
                        accumulatedGradients.put(s, gradArr);
                    } else if (!java.util.Arrays.equals(existing.shape(), gradArr.shape())) {
                        log.warn("Gradient shape changed mid-accumulation for '{}' in getOutputs: " +
                                 "expected {} got {}. Clearing accumulated state.",
                                s, java.util.Arrays.toString(existing.shape()),
                                java.util.Arrays.toString(gradArr.shape()));
                        accumulatedGradients = null;
                        accumulationStep = 0;
                        accumulatedGradients = new LinkedHashMap<>();
                        accumulatedGradients.put(s, gradArr);
                    } else {
                        existing.addi(gradArr);
                    }
                    // NOTE: applyUpdatersFromResults will detect that accumulatedGradients is
                    // already populated (from getOutputs) and will skip the results-based collection,
                    // then drive the accumulationStep counter and apply when the cycle completes.
                }
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
