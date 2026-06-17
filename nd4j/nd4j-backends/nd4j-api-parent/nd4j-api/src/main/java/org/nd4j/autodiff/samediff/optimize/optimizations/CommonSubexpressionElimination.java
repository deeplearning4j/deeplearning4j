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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.api.ops.ScalarOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Common Subexpression Elimination (CSE).
 *
 * Detects ops with identical signatures (same opName, same input variables in order,
 * same iArgs/tArgs/bArgs/dArgs/sArgs) and eliminates the duplicate by redirecting its
 * consumers to the output of the first (canonical) occurrence.
 *
 * This is one of the simplest and most universally applied compiler optimizations.
 * In ONNX-imported models it commonly deduplicates shape_of, cast, transpose, and
 * gather ops that get emitted multiple times for the same input.
 *
 * Placed after constant folding, algebraic simplification, and identity removal
 * (which may expose new duplicates) and before fusion passes (which benefit from
 * a smaller, deduplicated graph).
 */
@Slf4j
public class CommonSubexpressionElimination extends BaseOptimizerSet {

    public static class EliminateCommonSubexpressions implements Optimizer {

        private static final String RANDOM_OPS_PACKAGE = "org.nd4j.linalg.api.ops.random";

        /**
         * Signature -> canonical output variable names.
         * Persists across ops within one optimizer pass and across iterations.
         * Stale entries are detected by checking that canonical outputs still exist.
         */
        private final Map<String, List<String>> canonicalOutputs = new HashMap<>();

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            DifferentialFunction df = op.getOp();
            if (df == null) return false;

            // Skip non-deterministic ops — their outputs differ on each invocation
            if (df instanceof RandomOp) return false;
            String className = df.getClass().getName();
            if (className.startsWith(RANDOM_OPS_PACKAGE) && !className.endsWith(".Range")) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            List<String> outputs = op.getOutputsOfOp();
            if (inputs == null || outputs == null || outputs.isEmpty()) return false;

            // Ops with no inputs are either constants or generators — not CSE candidates
            if (inputs.isEmpty()) return false;

            String signature = computeSignature(df, inputs);

            List<String> canonical = canonicalOutputs.get(signature);
            if (canonical != null && canonical.size() == outputs.size()) {
                // Skip if this IS the canonical op (same output variable names).
                // This happens when the canonical's op is re-visited in a later iteration
                // or when the op map iteration processes the canonical after the duplicate.
                if (canonical.equals(outputs)) {
                    return false;
                }

                // Validate that canonical outputs still exist in the graph.
                // Prior passes may have removed the canonical op.
                if (!canonical.stream().allMatch(sd::hasVariable)) {
                    // Canonical was removed — this op becomes the new canonical
                    canonicalOutputs.put(signature, new ArrayList<>(outputs));
                    return false;
                }

                // Don't eliminate ops whose outputs are registered graph outputs
                List<String> graphOutputs = sd.outputs();
                if (graphOutputs != null && outputs.stream().anyMatch(graphOutputs::contains)) {
                    return false;
                }

                // Redirect all consumers of this op's outputs to the canonical outputs
                for (int i = 0; i < outputs.size(); i++) {
                    OptimizationUtils.replaceOpInputsWith(sd, helper, outputs.get(i), canonical.get(i));
                }

                // Remove the duplicate op and its output variables
                OptimizationUtils.removeOp(sd, helper, op.getName());
                outputs.forEach(out -> OptimizationUtils.removeVariable(sd, helper, out));

                log.debug("CSE: eliminated duplicate op {} ({}) — redirected {} -> canonical {}",
                        op.getName(), df.opName(), outputs, canonical);
                return true;
            }

            // First occurrence with this signature — register as canonical
            canonicalOutputs.put(signature, new ArrayList<>(outputs));
            return false;
        }

        /**
         * Compute a string signature that uniquely identifies an op's computation.
         * Two ops with the same signature produce identical results given the same inputs.
         */
        private String computeSignature(DifferentialFunction df, List<String> inputs) {
            StringBuilder sb = new StringBuilder(128);
            sb.append(df.opName());
            sb.append('|');

            // Input variable names in order (order matters for non-commutative ops like sub, div)
            sb.append(String.join(",", inputs));
            sb.append('|');

            // Op-specific arguments that affect the computation result
            if (df instanceof CustomOp) {
                CustomOp custom = (CustomOp) df;
                appendArgs(sb, custom.iArgs(), custom.tArgs(), custom.bArgs(), custom.dArgs(), custom.sArgs());
            } else if (df instanceof ScalarOp) {
                // Legacy scalar ops store the scalar value separately
                ScalarOp scalarOp = (ScalarOp) df;
                if (scalarOp.scalar() != null && scalarOp.scalar().isScalar()) {
                    sb.append("s=").append(scalarOp.scalar().getDouble(0));
                }
            }

            return sb.toString();
        }

        private void appendArgs(StringBuilder sb, long[] iArgs, double[] tArgs,
                                boolean[] bArgs, DataType[] dArgs, String[] sArgs) {
            if (iArgs != null && iArgs.length > 0) {
                sb.append('i');
                sb.append(Arrays.toString(iArgs));
            }
            if (tArgs != null && tArgs.length > 0) {
                sb.append('t');
                sb.append(Arrays.toString(tArgs));
            }
            if (bArgs != null && bArgs.length > 0) {
                sb.append('b');
                sb.append(Arrays.toString(bArgs));
            }
            if (dArgs != null && dArgs.length > 0) {
                sb.append('d');
                sb.append(Arrays.toString(dArgs));
            }
            if (sArgs != null && sArgs.length > 0) {
                sb.append('s');
                sb.append(Arrays.toString(sArgs));
            }
        }
    }
}
