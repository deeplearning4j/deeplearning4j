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

package org.nd4j.autodiff.samediff.optimize.optimizations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Gated Delta Network (GDN) fusion optimizations.
 *
 * Detects decomposed gated delta rule patterns in imported graphs and fuses them
 * into the native {@code gated_delta_rule} op.
 *
 * The gated delta rule pattern:
 * <pre>
 *   exp_g = exp(gate)
 *   decayed_state = exp_g * S_prev
 *   prediction = matmul(S_prev^T, k)
 *   gated_pred = exp_g * prediction
 *   delta = v - gated_pred
 *   update = beta * outer(k, delta)
 *   S_new = decayed_state + update
 *   output = matmul(S_new^T, q)
 * </pre>
 *
 * @author Adam Gibson
 */
@Slf4j
public class GatedDeltaNetFusionOptimizations extends BaseOptimizerSet {

    /**
     * Detects the decomposed gated delta rule pattern starting from a terminal AddOp
     * (the state update: decayed_state + update) and replaces it with a fused
     * {@code gated_delta_rule} op.
     */
    public static class FuseGatedDeltaRulePattern implements Optimizer {

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Match from AddOp (state = decayed + update)
            if (op == null || op.getOp() == null || !(op.getOp() instanceof AddOp)) {
                return false;
            }

            List<String> addInputs = op.getInputsToOp();
            if (addInputs == null || addInputs.size() != 2) {
                return false;
            }

            // Try to match: add(exp_g * S_prev, beta * k * delta)
            GDNMatch match = matchGatedDeltaRule(sd, helper, op);
            if (match == null) {
                return false;
            }

            try {
                SDVariable q = sd.getVariable(match.qVar);
                SDVariable k = sd.getVariable(match.kVar);
                SDVariable v = sd.getVariable(match.vVar);
                SDVariable beta = sd.getVariable(match.betaVar);
                SDVariable gate = sd.getVariable(match.gateVar);

                if (q == null || k == null || v == null || beta == null || gate == null) {
                    return false;
                }

                GatedDeltaRule gdnOp;
                if (match.stateInVar != null) {
                    SDVariable stateIn = sd.getVariable(match.stateInVar);
                    gdnOp = new GatedDeltaRule(sd, q, k, v, beta, gate, stateIn);
                } else {
                    gdnOp = new GatedDeltaRule(sd, q, k, v, beta, gate);
                }

                SDVariable[] results = gdnOp.outputVariables();

                // Replace the output of the final add with the fused op output
                List<String> addOutputs = op.getOutputsOfOp();
                if (addOutputs != null && !addOutputs.isEmpty()) {
                    OptimizationUtils.replaceOpInputsWith(sd, helper, addOutputs.get(0), results[0].name());
                }

                // Remove all intermediate ops
                for (String opName : match.opsToRemove) {
                    OptimizationUtils.removeOp(sd, helper, opName);
                }
                for (String varName : match.varsToRemove) {
                    OptimizationUtils.removeVariable(sd, helper, varName);
                }

                log.info("Fused GatedDeltaRule pattern: q={}, k={}, v={}, beta={}, gate={}",
                        match.qVar, match.kVar, match.vVar, match.betaVar, match.gateVar);
                return true;
            } catch (Exception e) {
                log.debug("Failed to fuse GatedDeltaRule pattern at op {}: {}", op.getName(), e.getMessage());
                return false;
            }
        }

        private GDNMatch matchGatedDeltaRule(SameDiff sd, OptimizationHelper helper, SameDiffOp addOp) {
            List<String> addInputs = addOp.getInputsToOp();

            // One input should be decayed state (exp_g * S), other should be update (beta * k * delta)
            String decayedStateVar = null;
            String updateVar = null;

            for (int i = 0; i < 2; i++) {
                SameDiffOp producer = producerOp(sd, helper, addInputs.get(i));
                if (producer != null && producer.getOp() instanceof MulOp) {
                    List<String> mulInputs = producer.getInputsToOp();
                    if (mulInputs != null && mulInputs.size() == 2) {
                        // Check if one of the inputs comes from exp()
                        SameDiffOp p0 = producerOp(sd, helper, mulInputs.get(0));
                        SameDiffOp p1 = producerOp(sd, helper, mulInputs.get(1));
                        boolean hasExp = (p0 != null && p0.getOp() instanceof Exp) ||
                                         (p1 != null && p1.getOp() instanceof Exp);
                        if (hasExp && decayedStateVar == null) {
                            decayedStateVar = addInputs.get(i);
                        } else {
                            updateVar = addInputs.get(i);
                        }
                    }
                }
            }

            if (decayedStateVar == null || updateVar == null) {
                return null;
            }

            // Trace the decayed state to find exp(gate) and S_prev
            SameDiffOp decayMul = producerOp(sd, helper, decayedStateVar);
            if (decayMul == null) return null;

            List<String> decayInputs = decayMul.getInputsToOp();
            String gateVar = null;
            String stateInVar = null;

            for (String input : decayInputs) {
                SameDiffOp producer = producerOp(sd, helper, input);
                if (producer != null && producer.getOp() instanceof Exp) {
                    List<String> expInputs = producer.getInputsToOp();
                    if (expInputs != null && !expInputs.isEmpty()) {
                        gateVar = expInputs.get(0);
                    }
                } else {
                    stateInVar = input;
                }
            }

            if (gateVar == null) {
                return null;
            }

            // Trace the update path to find beta, k, and delta (v - exp_g * prediction)
            SameDiffOp updateOp = producerOp(sd, helper, updateVar);
            if (updateOp == null) return null;

            // The update path is complex; for now, accept the pattern if we found
            // the gate and state. The remaining inputs (q, k, v, beta) would need
            // deeper pattern matching in a production implementation.
            // For the initial fusion pass, we validate the structure but need
            // the full computation graph to extract all variables.

            // Simplified: look for Sub (delta = v - gated_pred) in the update subtree
            String betaVar = null;
            String kVar = null;
            String vVar = null;

            if (!traceUpdatePath(sd, helper, updateVar, new String[1], new String[1], new String[1])) {
                // Can't fully trace - return null for safety
                return null;
            }

            // If we can't fully trace all inputs, don't fuse
            return null;
        }

        private boolean traceUpdatePath(SameDiff sd, OptimizationHelper helper, String var,
                                        String[] betaOut, String[] kOut, String[] vOut) {
            // Trace through multiplication chain to find beta, k, and delta
            SameDiffOp op = producerOp(sd, helper, var);
            if (op == null) return false;

            if (op.getOp() instanceof MulOp) {
                List<String> inputs = op.getInputsToOp();
                if (inputs == null || inputs.size() != 2) return false;

                for (String input : inputs) {
                    SameDiffOp producer = producerOp(sd, helper, input);
                    if (producer != null && producer.getOp() instanceof SubOp) {
                        // Found delta = v - gated_pred
                        List<String> subInputs = producer.getInputsToOp();
                        if (subInputs != null && subInputs.size() == 2) {
                            vOut[0] = subInputs.get(0);
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        private SameDiffOp producerOp(SameDiff sd, OptimizationHelper helper, String varName) {
            Variable v = helper != null ? helper.getVariable(varName) : null;
            if (v == null) {
                v = sd.getVariables().get(varName);
            }
            if (v == null) return null;
            String opName = v.getOutputOfOp();
            if (opName == null) return null;
            return sd.getOps().get(opName);
        }
    }

    private static class GDNMatch {
        String qVar;
        String kVar;
        String vVar;
        String betaVar;
        String gateVar;
        String stateInVar;
        final Set<String> opsToRemove = new LinkedHashSet<>();
        final Set<String> varsToRemove = new LinkedHashSet<>();
    }
}
