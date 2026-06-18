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

package org.nd4j.autodiff.samediff.execution;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.AdaBelief;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.AdaMax;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Appends updater ops (Adam, SGD, etc.) and weight-update ops directly into a
 * "grad" SameDiff graph so that the full training step — forward, backward,
 * optimizer, and weight update — executes inside a single DynamicShapePlan
 * without a Java-C++ round-trip after the backward pass.
 *
 * <h3>Graph structure added per trainable variable {@code w}</h3>
 * <pre>
 *   // Existing in grad graph after createGradFunction():
 *   w-grad  (ARRAY, output of backward op)
 *   w       (VARIABLE, the weight itself, as an external input)
 *
 *   // Added by this class for Adam:
 *   w__dsp_updater_m   (VARIABLE, Adam first-moment,  zero-initialized)
 *   w__dsp_updater_v   (VARIABLE, Adam second-moment, zero-initialized)
 *
 *   // Op: adam_updater(w-grad, w__dsp_updater_v, w__dsp_updater_m)
 *   //   → [w__dsp_upd_grad, w__dsp_updater_v_new, w__dsp_updater_m_new]
 *
 *   // Op: sub(w, w__dsp_upd_grad)           → w__dsp_weight_updated  (new weight value)
 *   //
 *   // After execution the caller (TrainingSession.applyFusedWeightUpdates) reads:
 *   //   w__dsp_weight_updated → copies into w's backing array
 *   //   adam_updater output[1] → copies into v's backing array (new second moment)
 *   //   adam_updater output[2] → copies into m's backing array (new first moment)
 * </pre>
 *
 * <h3>State variables and post-execution sync</h3>
 * Updater state arrays (Adam M/V, Nesterovs velocity, etc.) are registered as
 * {@code VARIABLE} type in the grad SameDiff. The DynamicShapePlanCompiler picks
 * them up as VARIABLE external inputs and hydrates them from the SDVariable's
 * backing array before each execution. The new-state output variables produced
 * by the updater ops are recorded in {@link AppendResult#stateVarToNewOutput};
 * after each DSP execution the caller copies those tensors back into the state
 * SDVariable backing arrays so the next execution reads the correct values.
 *
 * <h3>Iteration counter</h3>
 * Adam and similar updaters need the iteration count as an {@code iArg}. The
 * iteration is baked in at graph-append time. The caller must call
 * {@link #appendUpdaterOps} again (and recompile the DSP plan) when the iteration
 * counter changes significantly enough to matter. For most training loops this
 * means recompiling once per epoch or every N steps; a simple approach is to
 * recompile at iteration 0 only and rely on the tArg frozen in the slot. Future
 * work can expose iteration as a placeholder input.
 *
 * <h3>Supported updaters</h3>
 * <ul>
 *   <li>SGD — no state</li>
 *   <li>Nesterovs — 1 state (velocity V)</li>
 *   <li>Adam — 2 state (M, V)</li>
 *   <li>AdaMax — 2 state (M, V)</li>
 *   <li>AdaBelief — 2 state (M, V)</li>
 *   <li>Nadam — 2 state (M, V)</li>
 *   <li>AMSGrad — 3 state (M, V, H)</li>
 *   <li>AdaDelta — 2 state (Msg, Msdx)</li>
 *   <li>AdaGrad — 1 state (G)</li>
 *   <li>RmsProp — 1 state (G)</li>
 * </ul>
 *
 * Unsupported updaters produce a warning and the variable is skipped (caller falls
 * back to the Java-side updater for that variable).
 */
@Slf4j
public class UpdaterOpsAppender {

    /**
     * Infix used for updater-state variable names so they are recognisable in the graph.
     * Format: {@code <varName>__dsp_updater_<stateKey>}
     */
    public static final String STATE_VAR_PREFIX = "__dsp_updater_";

    /**
     * Suffix on the final weight-assign output variable.
     * Format: {@code <varName>__dsp_weight_updated}
     */
    public static final String WEIGHT_UPDATED_SUFFIX = "__dsp_weight_updated";

    /**
     * Result of appending updater ops to the grad graph.
     */
    public static class AppendResult {
        /**
         * Map from trainable variable name → name of the new-weight output variable
         * in the grad graph (e.g. {@code w__dsp_w_delta} before the rename, or
         * {@code w__dsp_weight_updated}).
         * The DSP executor must be asked to produce these outputs; after execution the
         * caller copies the resulting tensor into the weight SDVariable's backing array.
         */
        public final Map<String, String> varToWeightUpdatedOutput;

        /**
         * Map from state-variable name → name of the plan output variable that carries
         * the updated state for that slot. After each DSP execution the caller reads
         * these outputs and copies the tensors back into the state SDVariable's backing
         * arrays so the next iteration sees the correct moment estimates.
         *
         * <p>Example: {@code "w__dsp_updater_v" → "w__dsp_updater_v_new"}</p>
         */
        public final Map<String, String> stateVarToNewOutput;

        /**
         * Names of all updater-state variables added as VARIABLE type. These appear in
         * the plan's external input list and are hydrated before each execution.
         */
        public final List<String> updaterStateVarNames;

        /**
         * Variables that were skipped (unsupported updater type, missing gradient, etc.).
         */
        public final List<String> skippedVars;

        public AppendResult(Map<String, String> varToWeightUpdatedOutput,
                            Map<String, String> stateVarToNewOutput,
                            List<String> updaterStateVarNames,
                            List<String> skippedVars) {
            this.varToWeightUpdatedOutput = varToWeightUpdatedOutput;
            this.stateVarToNewOutput = stateVarToNewOutput;
            this.updaterStateVarNames = updaterStateVarNames;
            this.skippedVars = skippedVars;
        }
    }

    private UpdaterOpsAppender() {}

    /**
     * Append updater ops and weight-update ops to the given grad SameDiff graph.
     *
     * <p>This method is intended to be called once, after
     * {@code sd.createGradFunction()} and {@code sd.initializeTraining()}.
     * The caller must recompile the DSP plan after calling this method because the
     * graph structure has changed.</p>
     *
     * @param gradSd     The "grad" SameDiff instance (returned by {@code sd.getFunction("grad")}).
     * @param config     The training configuration — used to resolve per-variable updater config.
     * @param updaterMap Already-initialised updater map (variable name → GradientUpdater).
     *                   The GradientUpdater's state arrays are registered directly as graph variables.
     * @param iteration  Current training iteration. Baked into ops that need it as iArg.
     * @param minimize   True for loss minimization (subtract updated grad), false for maximization.
     * @return           AppendResult describing what was added and what was skipped.
     */
    public static AppendResult appendUpdaterOps(SameDiff gradSd,
                                                TrainingConfig config,
                                                Map<String, GradientUpdater> updaterMap,
                                                int iteration,
                                                boolean minimize) {
        Map<String, String> varToUpdatedOutput = new LinkedHashMap<>();
        Map<String, String> stateVarToNewOutput = new LinkedHashMap<>();
        List<String> stateVarNames = new ArrayList<>();
        List<String> skipped = new ArrayList<>();

        for (Map.Entry<String, GradientUpdater> entry : updaterMap.entrySet()) {
            String varName = entry.getKey();
            GradientUpdater gu = entry.getValue();

            // The gradient variable is named "<varName>-grad" by createGradFunction convention.
            String gradVarName = varName + "-grad";
            if (!gradSd.hasVariable(gradVarName)) {
                log.debug("UpdaterOpsAppender: no grad variable '{}' found for '{}', skipping",
                        gradVarName, varName);
                skipped.add(varName);
                continue;
            }

            // Weight variable must exist in the grad graph (copied there by invokeGraphOn).
            if (!gradSd.hasVariable(varName)) {
                log.warn("UpdaterOpsAppender: weight variable '{}' not in grad graph, skipping", varName);
                skipped.add(varName);
                continue;
            }

            IUpdater updaterConfig = config.getUpdaterForVariable(varName);
            if (updaterConfig == null) {
                log.warn("UpdaterOpsAppender: no updater config for '{}', skipping", varName);
                skipped.add(varName);
                continue;
            }

            try {
                String updatedOutputName = appendSingleUpdater(
                        gradSd, varName, gradVarName, updaterConfig, gu,
                        stateVarNames, stateVarToNewOutput, iteration, minimize);
                if (updatedOutputName != null) {
                    varToUpdatedOutput.put(varName, updatedOutputName);
                    log.debug("UpdaterOpsAppender: '{}' → weight output='{}'", varName, updatedOutputName);
                } else {
                    skipped.add(varName);
                }
            } catch (Exception e) {
                log.warn("UpdaterOpsAppender: failed for '{}': {}", varName, e.getMessage(), e);
                skipped.add(varName);
            }
        }

        log.info("UpdaterOpsAppender: appended updater ops for {}/{} variables " +
                 "({} skipped, {} state vars)",
                varToUpdatedOutput.size(), updaterMap.size(), skipped.size(), stateVarNames.size());

        return new AppendResult(varToUpdatedOutput, stateVarToNewOutput, stateVarNames, skipped);
    }

    // -----------------------------------------------------------------------
    // Dispatcher
    // -----------------------------------------------------------------------

    /**
     * Append updater and weight-update ops for one variable.
     *
     * @param stateVarToNewOutput Populated in-place: maps state-var name → plan output name
     *                            carrying the new state value after this updater runs.
     * @return Name of the new-weight output variable, or null if skipped.
     */
    private static String appendSingleUpdater(SameDiff sd,
                                              String varName,
                                              String gradVarName,
                                              IUpdater updaterConfig,
                                              GradientUpdater gu,
                                              List<String> stateVarNames,
                                              Map<String, String> stateVarToNewOutput,
                                              int iteration,
                                              boolean minimize) {
        SDVariable gradVar = sd.getVariable(gradVarName);
        SDVariable weightVar = sd.getVariable(varName);
        DataType dtype = weightVar.dataType();

        if (!dtype.isFPType()) {
            log.debug("UpdaterOpsAppender: '{}' is not FP ({}), skipping", varName, dtype);
            return null;
        }

        if (updaterConfig instanceof Sgd) {
            return appendSgd(sd, varName, gradVar, weightVar, (Sgd) updaterConfig, iteration, minimize);
        } else if (updaterConfig instanceof Nesterovs) {
            return appendNesterovs(sd, varName, gradVar, weightVar,
                    (Nesterovs) updaterConfig, gu, stateVarNames, stateVarToNewOutput, iteration, minimize);
        } else if (updaterConfig instanceof Adam) {
            return appendAdam(sd, varName, gradVar, weightVar,
                    (Adam) updaterConfig, gu, stateVarNames, stateVarToNewOutput, iteration, minimize);
        } else if (updaterConfig instanceof AdaMax) {
            return appendAdaMax(sd, varName, gradVar, weightVar,
                    (AdaMax) updaterConfig, gu, stateVarNames, stateVarToNewOutput, iteration, minimize);
        } else if (updaterConfig instanceof AdaBelief) {
            return appendAdaBelief(sd, varName, gradVar, weightVar,
                    (AdaBelief) updaterConfig, gu, stateVarNames, stateVarToNewOutput, iteration, minimize);
        } else if (updaterConfig instanceof Nadam) {
            return appendNadam(sd, varName, gradVar, weightVar,
                    (Nadam) updaterConfig, gu, stateVarNames, stateVarToNewOutput, iteration, minimize);
        } else if (updaterConfig instanceof AMSGrad) {
            return appendAmsGrad(sd, varName, gradVar, weightVar,
                    (AMSGrad) updaterConfig, gu, stateVarNames, stateVarToNewOutput, iteration, minimize);
        } else if (updaterConfig instanceof AdaDelta) {
            return appendAdaDelta(sd, varName, gradVar, weightVar,
                    (AdaDelta) updaterConfig, gu, stateVarNames, stateVarToNewOutput, iteration, minimize);
        } else if (updaterConfig instanceof AdaGrad) {
            return appendAdaGrad(sd, varName, gradVar, weightVar,
                    (AdaGrad) updaterConfig, gu, stateVarNames, stateVarToNewOutput, iteration, minimize);
        } else if (updaterConfig instanceof RmsProp) {
            return appendRmsProp(sd, varName, gradVar, weightVar,
                    (RmsProp) updaterConfig, gu, stateVarNames, stateVarToNewOutput, iteration, minimize);
        } else {
            log.warn("UpdaterOpsAppender: unsupported updater '{}' for '{}', skipping",
                    updaterConfig.getClass().getSimpleName(), varName);
            return null;
        }
    }

    // -----------------------------------------------------------------------
    // SGD — no state arrays, 1 tArg (lr)
    // Signature: sgd_updater(grad) → [updated_grad]
    // -----------------------------------------------------------------------
    private static String appendSgd(SameDiff sd, String varName,
                                    SDVariable gradVar, SDVariable weightVar,
                                    Sgd config, int iteration, boolean minimize) {
        double lr = config.getLearningRate(iteration, 0);
        // sgd_updater(grad) → updated_grad  (scales by lr)
        SDVariable[] updOut = addCustomOp(sd, "sgd_updater",
                new SDVariable[]{gradVar},
                new double[]{lr},
                new long[0],
                varName + STATE_VAR_PREFIX + "sgd_upd",
                1);
        SDVariable updatedGrad = updOut[0];
        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // Nesterovs — 1 state (V), 2 tArgs (lr, momentum)
    // Signature: nesterovs_updater(grad, v) → [updated_grad, new_v]
    // -----------------------------------------------------------------------
    private static String appendNesterovs(SameDiff sd, String varName,
                                          SDVariable gradVar, SDVariable weightVar,
                                          Nesterovs config, GradientUpdater gu,
                                          List<String> stateVarNames,
                                          Map<String, String> stateVarToNewOutput,
                                          int iteration, boolean minimize) {
        double lr = config.getLearningRate(iteration, 0);
        double momentum = config.getMomentum();
        String stateVName = varName + STATE_VAR_PREFIX + "v";

        SDVariable stateV = getOrCreateStateVar(sd, stateVName, weightVar, gu, "V", stateVarNames);

        SDVariable[] updOut = addCustomOp(sd, "nesterovs_updater",
                new SDVariable[]{gradVar, stateV},
                new double[]{lr, momentum},
                new long[0],
                varName + STATE_VAR_PREFIX + "nest_upd",
                2);
        SDVariable updatedGrad = updOut[0];
        SDVariable newV = updOut[1];
        // Record that after execution the caller should copy newV into stateV's backing array.
        stateVarToNewOutput.put(stateVName, newV.name());

        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // Adam — 2 state (V=second-moment, M=first-moment), 4 tArgs, 1 iArg
    // C++ signature: adam_updater(grad, initStateU=V, initStateM=M) → [updated_grad, new_V, new_M]
    // -----------------------------------------------------------------------
    private static String appendAdam(SameDiff sd, String varName,
                                     SDVariable gradVar, SDVariable weightVar,
                                     Adam config, GradientUpdater gu,
                                     List<String> stateVarNames,
                                     Map<String, String> stateVarToNewOutput,
                                     int iteration, boolean minimize) {
        double lr = config.getLearningRate(iteration, 0);
        double beta1 = config.getBeta1();
        double beta2 = config.getBeta2();
        double epsilon = config.getEpsilon();
        String stateVName = varName + STATE_VAR_PREFIX + "v";
        String stateMName = varName + STATE_VAR_PREFIX + "m";

        // V = second moment (stateU in C++), M = first moment (stateM in C++)
        SDVariable stateV = getOrCreateStateVar(sd, stateVName, weightVar, gu, "V", stateVarNames);
        SDVariable stateM = getOrCreateStateVar(sd, stateMName, weightVar, gu, "M", stateVarNames);

        SDVariable[] updOut = addCustomOp(sd, "adam_updater",
                new SDVariable[]{gradVar, stateV, stateM},
                new double[]{lr, beta1, beta2, epsilon},
                new long[]{iteration},
                varName + STATE_VAR_PREFIX + "adam_upd",
                3);
        SDVariable updatedGrad = updOut[0];
        stateVarToNewOutput.put(stateVName, updOut[1].name());
        stateVarToNewOutput.put(stateMName, updOut[2].name());

        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // AdaMax — 2 state (V, M), 4 tArgs, 1 iArg
    // -----------------------------------------------------------------------
    private static String appendAdaMax(SameDiff sd, String varName,
                                       SDVariable gradVar, SDVariable weightVar,
                                       AdaMax config, GradientUpdater gu,
                                       List<String> stateVarNames,
                                       Map<String, String> stateVarToNewOutput,
                                       int iteration, boolean minimize) {
        double lr = config.getLearningRate(iteration, 0);
        double beta1 = config.getBeta1();
        double beta2 = config.getBeta2();
        double epsilon = config.getEpsilon();
        String stateVName = varName + STATE_VAR_PREFIX + "v";
        String stateMName = varName + STATE_VAR_PREFIX + "m";

        SDVariable stateV = getOrCreateStateVar(sd, stateVName, weightVar, gu, "V", stateVarNames);
        SDVariable stateM = getOrCreateStateVar(sd, stateMName, weightVar, gu, "M", stateVarNames);

        SDVariable[] updOut = addCustomOp(sd, "ada_max_updater",
                new SDVariable[]{gradVar, stateV, stateM},
                new double[]{lr, beta1, beta2, epsilon},
                new long[]{iteration},
                varName + STATE_VAR_PREFIX + "adamax_upd",
                3);
        SDVariable updatedGrad = updOut[0];
        stateVarToNewOutput.put(stateVName, updOut[1].name());
        stateVarToNewOutput.put(stateMName, updOut[2].name());

        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // AdaBelief — 2 state (V, M), 4 tArgs, 1 iArg
    // -----------------------------------------------------------------------
    private static String appendAdaBelief(SameDiff sd, String varName,
                                          SDVariable gradVar, SDVariable weightVar,
                                          AdaBelief config, GradientUpdater gu,
                                          List<String> stateVarNames,
                                          Map<String, String> stateVarToNewOutput,
                                          int iteration, boolean minimize) {
        double lr = config.getLearningRate(iteration, 0);
        double beta1 = config.getBeta1();
        double beta2 = config.getBeta2();
        double epsilon = config.getEpsilon();
        String stateVName = varName + STATE_VAR_PREFIX + "v";
        String stateMName = varName + STATE_VAR_PREFIX + "m";

        SDVariable stateV = getOrCreateStateVar(sd, stateVName, weightVar, gu, "V", stateVarNames);
        SDVariable stateM = getOrCreateStateVar(sd, stateMName, weightVar, gu, "M", stateVarNames);

        SDVariable[] updOut = addCustomOp(sd, "adabelief_updater",
                new SDVariable[]{gradVar, stateV, stateM},
                new double[]{lr, beta1, beta2, epsilon},
                new long[]{iteration},
                varName + STATE_VAR_PREFIX + "adabelief_upd",
                3);
        SDVariable updatedGrad = updOut[0];
        stateVarToNewOutput.put(stateVName, updOut[1].name());
        stateVarToNewOutput.put(stateMName, updOut[2].name());

        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // Nadam — 2 state (V, M), 4 tArgs, 1 iArg
    // -----------------------------------------------------------------------
    private static String appendNadam(SameDiff sd, String varName,
                                      SDVariable gradVar, SDVariable weightVar,
                                      Nadam config, GradientUpdater gu,
                                      List<String> stateVarNames,
                                      Map<String, String> stateVarToNewOutput,
                                      int iteration, boolean minimize) {
        double lr = config.getLearningRate(iteration, 0);
        double beta1 = config.getBeta1();
        double beta2 = config.getBeta2();
        double epsilon = config.getEpsilon();
        String stateVName = varName + STATE_VAR_PREFIX + "v";
        String stateMName = varName + STATE_VAR_PREFIX + "m";

        SDVariable stateV = getOrCreateStateVar(sd, stateVName, weightVar, gu, "V", stateVarNames);
        SDVariable stateM = getOrCreateStateVar(sd, stateMName, weightVar, gu, "M", stateVarNames);

        SDVariable[] updOut = addCustomOp(sd, "nadam_updater",
                new SDVariable[]{gradVar, stateV, stateM},
                new double[]{lr, beta1, beta2, epsilon},
                new long[]{iteration},
                varName + STATE_VAR_PREFIX + "nadam_upd",
                3);
        SDVariable updatedGrad = updOut[0];
        stateVarToNewOutput.put(stateVName, updOut[1].name());
        stateVarToNewOutput.put(stateMName, updOut[2].name());

        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // AMSGrad — 3 state (V, M, H), 4 tArgs, 1 iArg
    // -----------------------------------------------------------------------
    private static String appendAmsGrad(SameDiff sd, String varName,
                                        SDVariable gradVar, SDVariable weightVar,
                                        AMSGrad config, GradientUpdater gu,
                                        List<String> stateVarNames,
                                        Map<String, String> stateVarToNewOutput,
                                        int iteration, boolean minimize) {
        double lr = config.getLearningRate(iteration, 0);
        double beta1 = config.getBeta1();
        double beta2 = config.getBeta2();
        double epsilon = config.getEpsilon();
        String stateVName = varName + STATE_VAR_PREFIX + "v";
        String stateMName = varName + STATE_VAR_PREFIX + "m";
        String stateHName = varName + STATE_VAR_PREFIX + "h";

        SDVariable stateV = getOrCreateStateVar(sd, stateVName, weightVar, gu, "V", stateVarNames);
        SDVariable stateM = getOrCreateStateVar(sd, stateMName, weightVar, gu, "M", stateVarNames);
        SDVariable stateH = getOrCreateStateVar(sd, stateHName, weightVar, gu, "H", stateVarNames);

        SDVariable[] updOut = addCustomOp(sd, "ams_grad_updater",
                new SDVariable[]{gradVar, stateV, stateM, stateH},
                new double[]{lr, beta1, beta2, epsilon},
                new long[]{iteration},
                varName + STATE_VAR_PREFIX + "amsgrad_upd",
                4);
        SDVariable updatedGrad = updOut[0];
        stateVarToNewOutput.put(stateVName, updOut[1].name());
        stateVarToNewOutput.put(stateMName, updOut[2].name());
        stateVarToNewOutput.put(stateHName, updOut[3].name());

        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // AdaDelta — 2 state (Msg, Msdx), 2 tArgs (rho, epsilon)
    // -----------------------------------------------------------------------
    private static String appendAdaDelta(SameDiff sd, String varName,
                                         SDVariable gradVar, SDVariable weightVar,
                                         AdaDelta config, GradientUpdater gu,
                                         List<String> stateVarNames,
                                         Map<String, String> stateVarToNewOutput,
                                         int iteration, boolean minimize) {
        double rho = config.getRho();
        double epsilon = config.getEpsilon();
        String stateMsgName = varName + STATE_VAR_PREFIX + "msg";
        String stateMsdxName = varName + STATE_VAR_PREFIX + "msdx";

        SDVariable stateMsg = getOrCreateStateVar(sd, stateMsgName, weightVar, gu, "Msg", stateVarNames);
        SDVariable stateMsdx = getOrCreateStateVar(sd, stateMsdxName, weightVar, gu, "Msdx", stateVarNames);

        SDVariable[] updOut = addCustomOp(sd, "ada_delta_updater",
                new SDVariable[]{gradVar, stateMsg, stateMsdx},
                new double[]{rho, epsilon},
                new long[0],
                varName + STATE_VAR_PREFIX + "adadelta_upd",
                3);
        SDVariable updatedGrad = updOut[0];
        stateVarToNewOutput.put(stateMsgName, updOut[1].name());
        stateVarToNewOutput.put(stateMsdxName, updOut[2].name());

        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // AdaGrad — 1 state (G), 2 tArgs (lr, epsilon)
    // -----------------------------------------------------------------------
    private static String appendAdaGrad(SameDiff sd, String varName,
                                        SDVariable gradVar, SDVariable weightVar,
                                        AdaGrad config, GradientUpdater gu,
                                        List<String> stateVarNames,
                                        Map<String, String> stateVarToNewOutput,
                                        int iteration, boolean minimize) {
        double lr = config.getLearningRate(iteration, 0);
        double epsilon = config.getEpsilon();
        String stateGName = varName + STATE_VAR_PREFIX + "g";

        SDVariable stateG = getOrCreateStateVar(sd, stateGName, weightVar, gu, "G", stateVarNames);

        SDVariable[] updOut = addCustomOp(sd, "ada_grad_updater",
                new SDVariable[]{gradVar, stateG},
                new double[]{lr, epsilon},
                new long[0],
                varName + STATE_VAR_PREFIX + "adagrad_upd",
                2);
        SDVariable updatedGrad = updOut[0];
        stateVarToNewOutput.put(stateGName, updOut[1].name());

        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // RmsProp — 1 state (G), 3 tArgs (lr, decay, epsilon)
    // -----------------------------------------------------------------------
    private static String appendRmsProp(SameDiff sd, String varName,
                                        SDVariable gradVar, SDVariable weightVar,
                                        RmsProp config, GradientUpdater gu,
                                        List<String> stateVarNames,
                                        Map<String, String> stateVarToNewOutput,
                                        int iteration, boolean minimize) {
        double lr = config.getLearningRate(iteration, 0);
        double decay = config.getRmsDecay();
        double epsilon = config.getEpsilon();
        String stateGName = varName + STATE_VAR_PREFIX + "g";

        SDVariable stateG = getOrCreateStateVar(sd, stateGName, weightVar, gu, "G", stateVarNames);

        SDVariable[] updOut = addCustomOp(sd, "rms_prop_updater",
                new SDVariable[]{gradVar, stateG},
                new double[]{lr, decay, epsilon},
                new long[0],
                varName + STATE_VAR_PREFIX + "rmsprop_upd",
                2);
        SDVariable updatedGrad = updOut[0];
        stateVarToNewOutput.put(stateGName, updOut[1].name());

        return appendWeightUpdate(sd, varName, weightVar, updatedGrad, minimize);
    }

    // -----------------------------------------------------------------------
    // Low-level helpers
    // -----------------------------------------------------------------------

    /**
     * Add a custom op to the SameDiff graph using the {@code sd.dynamic()} API.
     *
     * <p>This is the correct programmatic way to add a named custom op to a SameDiff
     * graph. It delegates to {@link SameDiff#dynamic(String, List, List, List, List, List, List)}
     * which handles op registration, input wiring, argument storage, and output variable
     * creation in one call. The first argument to {@code sd.dynamic()} is the <em>op name</em>
     * (used to look up the op class via {@code DifferentialFunctionClassHolder}), not a base
     * name for outputs.</p>
     *
     * @param sd         SameDiff instance.
     * @param opName     Registered C++ op name (e.g. "adam_updater"). Must match the value
     *                   returned by the op's {@code opName()} method.
     * @param inputs     Input SDVariable array.
     * @param tArgs      Floating-point arguments (lr, beta1, etc.).
     * @param iArgs      Integer arguments (iteration, etc.). Cast to int internally.
     * @param baseName   Unused parameter — kept for signature clarity. Outputs are auto-named.
     * @param numOutputs Unused parameter — actual output count is from the C++ descriptor.
     * @return Array of output SDVariables (length determined by the C++ op descriptor).
     */
    private static SDVariable[] addCustomOp(SameDiff sd,
                                            String opName,
                                            SDVariable[] inputs,
                                            double[] tArgs,
                                            long[] iArgs,
                                            String baseName,
                                            int numOutputs) {
        List<SDVariable> inputList = Arrays.asList(inputs);

        List<Long> iArgsList = new ArrayList<>(iArgs.length);
        for (long v : iArgs) iArgsList.add(v);

        List<Double> tArgsList = new ArrayList<>(tArgs.length);
        for (double v : tArgs) tArgsList.add(v);

        // sd.dynamic(opName, ...) looks up the op class by opName in OP_NAME_MAP,
        // creates a new instance, wires inputs via addArgsFor(), stores all args,
        // and calls outputVariables() to generate the output SDVariables.
        // The opName is the registered C++ op name (e.g. "adam_updater").
        SDVariable[] out = sd.dynamic(opName, inputList, iArgsList, tArgsList,
                java.util.Collections.emptyList(),
                java.util.Collections.emptyList(),
                java.util.Collections.emptyList());

        return out;
    }

    /**
     * Get an existing updater-state VARIABLE from the grad graph, or create it if absent.
     *
     * <p>The backing INDArray is pulled from the GradientUpdater's state map
     * (which was already populated by {@code setStateViewArray} in
     * {@code SameDiff.initializeTraining()}). The array is registered as a VARIABLE
     * so the plan treats it as a mutable external input that persists across steps.</p>
     *
     * @param sd          The grad SameDiff instance.
     * @param stateVarName Unique name for the state variable in the graph.
     * @param likeVar     Reference weight variable — used only as dtype fallback.
     * @param gu          GradientUpdater whose {@code getState()} map provides the backing array.
     * @param stateKey    Key in {@code gu.getState()} (e.g. "M", "V", "G").
     * @param stateVarNames Accumulation list — the new var name is appended here if created.
     * @return The SDVariable representing the state array in the grad graph.
     */
    private static SDVariable getOrCreateStateVar(SameDiff sd,
                                                  String stateVarName,
                                                  SDVariable likeVar,
                                                  GradientUpdater gu,
                                                  String stateKey,
                                                  List<String> stateVarNames) {
        if (sd.hasVariable(stateVarName)) {
            return sd.getVariable(stateVarName);
        }

        Map<String, INDArray> state = gu.getState();
        INDArray stateArr = (state != null) ? state.get(stateKey) : null;

        SDVariable stateVar;
        if (stateArr != null) {
            // Register the live state array from the updater directly.
            // var(name, array) creates a VARIABLE-type SDVariable backed by this array.
            stateVar = sd.var(stateVarName, stateArr);
        } else {
            // Fallback: SGD and similar zero-state updaters. Create a zero array with
            // the weight's shape so the plan slot has a valid backing array.
            long[] shape = likeVar.getShape();
            INDArray zeroArr;
            if (shape != null && shape.length > 0) {
                zeroArr = Nd4j.zeros(likeVar.dataType(), shape);
            } else {
                zeroArr = Nd4j.scalar(likeVar.dataType(), 0.0);
            }
            stateVar = sd.var(stateVarName, zeroArr);
        }

        stateVarNames.add(stateVarName);
        return stateVar;
    }

    /**
     * Add a subtract (or add) op to compute the new weight value.
     *
     * <pre>
     *   newWeight = w - updatedGrad   (or w + updatedGrad if !minimize)
     * </pre>
     *
     * The name of the new-weight output variable is returned and recorded in
     * {@code AppendResult.varToWeightUpdatedOutput}. After DSP execution the
     * caller reads this output from the results map and copies it into the
     * weight SDVariable's backing array (no in-graph assign op needed).
     *
     * @param sd           SameDiff grad instance.
     * @param varName      Weight variable name (used for naming outputs).
     * @param weightVar    The weight SDVariable.
     * @param updatedGrad  Output of the updater op (the scaled gradient update to apply).
     * @param minimize     True → subtract; false → add.
     * @return Name of the new-weight output variable, or null on error.
     */
    private static String appendWeightUpdate(SameDiff sd, String varName,
                                             SDVariable weightVar,
                                             SDVariable updatedGrad,
                                             boolean minimize) {
        // Compute new weight value: newWeight = weight ± updatedGrad
        SDVariable newWeight;
        if (minimize) {
            newWeight = sd.math().sub(weightVar, updatedGrad);
        } else {
            newWeight = sd.math().add(weightVar, updatedGrad);
        }
        // Rename to a stable, recognisable name for the results map lookup.
        String newWeightName = varName + WEIGHT_UPDATED_SUFFIX;
        try {
            newWeight.rename(newWeightName);
        } catch (Exception e) {
            // Name collision — use whatever name the var got.
            newWeightName = newWeight.name();
            log.debug("UpdaterOpsAppender: could not rename new-weight output for '{}', using '{}'",
                    varName, newWeightName);
        }
        return newWeightName;
    }

    // -----------------------------------------------------------------------
    // Inspection helpers
    // -----------------------------------------------------------------------

    /**
     * Return true if the grad graph already has updater ops appended (idempotency check).
     */
    public static boolean hasUpdaterOps(SameDiff gradSd) {
        for (String name : gradSd.variableNames()) {
            if (name.contains(STATE_VAR_PREFIX)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Return all updater-state variable names in the grad graph.
     * Useful when reloading a checkpointed grad graph to reconstruct state.
     */
    public static List<String> findUpdaterStateVarNames(SameDiff gradSd) {
        List<String> result = new ArrayList<>();
        for (String name : gradSd.variableNames()) {
            if (name.contains(STATE_VAR_PREFIX) && !name.contains(WEIGHT_UPDATED_SUFFIX)) {
                result.add(name);
            }
        }
        return result;
    }
}
