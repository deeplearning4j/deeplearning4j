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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

@Slf4j
public class OptimizationUtils {

    private OptimizationUtils(){ }

    /**
     * Replace op inputs with a new variable using fast O(1) lookups via helper.
     */
    public static void replaceOpInputsWith(SameDiff sd, OptimizationHelper helper, @NonNull String replaceInput, @NonNull String newInput){
        if(replaceInput.equals(newInput))
            return;

        //Update op input structure: Replace all instances replaceInput->X with newInput->X
        Collection<SameDiffOp> ops = sd.getOps().values();
        for(SameDiffOp o : ops){
            List<String> l = o.getInputsToOp();
            while(l != null && l.contains(replaceInput)){
                int idx = l.indexOf(replaceInput);
                l.set(idx, newInput);
            }
        }

        //Update variable structure - use fast O(1) lookup via helper, with fallback to graph
        Variable v = helper != null ? helper.getVariable(replaceInput) : null;
        if (v == null) {
            v = sd.getVariables().get(replaceInput);
        }

        Variable v2 = helper != null ? helper.getVariable(newInput) : null;
        if (v2 == null) {
            // New variables created during optimization won't be in the cache
            v2 = sd.getVariables().get(newInput);
            // Add to cache for future lookups
            if (helper != null && v2 != null) {
                helper.updateVariable(newInput, v2);
            }
        }

        // MERGE: v2 keeps its existing consumers AND inherits v's consumers.
        // Previous code REPLACED v2's consumers, which lost references when v2 was already used by other ops.
        if (v != null && v2 != null) {
            List<String> merged = v2.getInputsForOp() != null
                    ? new ArrayList<>(v2.getInputsForOp()) : new ArrayList<>();
            if (v.getInputsForOp() != null) {
                for (String opName : v.getInputsForOp()) {
                    if (!merged.contains(opName)) {
                        merged.add(opName);
                    }
                }
            }
            v2.setInputsForOp(merged);
            v.setInputsForOp(new ArrayList<>());
        }
    }

    /**
     * @deprecated Use {@link #replaceOpInputsWith(SameDiff, OptimizationHelper, String, String)} for better performance
     */
    @Deprecated
    public static void replaceOpInputsWith(SameDiff sd, @NonNull String replaceInput, @NonNull String newInput){
        replaceOpInputsWith(sd, null, replaceInput, newInput);
    }

    /**
     * Remove an op from the graph using fast O(1) lookups via helper.
     *
     * <p>Refuses to remove any op whose output variables are registered graph outputs
     * (e.g. KV cache outputs like {@code k_rope_0}, {@code v_heads_0}, or ONNX
     * {@code present_*} outputs). Removing such an op would orphan the output variable
     * — its producing op would be gone but the variable would still exist in the graph,
     * causing silent wrong results at runtime.</p>
     */
    public static void removeOp(@NonNull SameDiff sd, OptimizationHelper helper, @NonNull String opToRemove){
        // Guard: never remove an op that produces a registered graph output.
        // KV cache outputs (k_rope_N, v_heads_N, present_*) are registered in sd.outputs()
        // and must remain reachable. Removing their producing op orphans the variable.
        SameDiffOp opCheck = sd.getOps().get(opToRemove);
        if (opCheck != null && opCheck.getOutputsOfOp() != null) {
            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null && !graphOutputs.isEmpty()) {
                for (String outVar : opCheck.getOutputsOfOp()) {
                    if (graphOutputs.contains(outVar)) {
                        log.warn("Refusing to remove op '{}' — its output '{}' is a registered graph output",
                                opToRemove, outVar);
                        return;
                    }
                }
            }
        }

        SameDiffOp op = sd.getOps().remove(opToRemove);
        if (op == null) {
            return; // Op already removed or doesn't exist
        }
        if (helper != null) {
            helper.updateOp(opToRemove, null);
        }
        List<String> inputs = op.getInputsToOp();
        if (inputs == null) {
            return;
        }
        for(String s : inputs){
            // Use fast O(1) lookup via helper, with fallback to graph
            Variable v = helper != null ? helper.getVariable(s) : null;
            if (v == null) {
                v = sd.getVariables().get(s);
            }
            if (v != null && v.getInputsForOp() != null) {
                // removeAll to handle ops that use the same variable as multiple inputs
                // (e.g., x*x). List.remove(Object) only removes the first occurrence,
                // leaving a stale entry that causes DSP to see a phantom consumer.
                while (v.getInputsForOp().remove(op.getName())) { }
            }
        }
    }

    /**
     * @deprecated Use {@link #removeOp(SameDiff, OptimizationHelper, String)} for better performance
     */
    @Deprecated
    public static void removeOp(@NonNull SameDiff sd, @NonNull String opToRemove){
        removeOp(sd, null, opToRemove);
    }

    /**
     * Remove a variable from the graph using helper to update caches.
     */
    public static void removeVariable(@NonNull SameDiff sd, OptimizationHelper helper, @NonNull String varToRemove){
        // Never remove a variable that is a registered graph output — the native plan,
        // output collection, and downstream consumers all rely on the name existing.
        List<String> outputs = sd.outputs();
        if (outputs != null && outputs.contains(varToRemove)) {
            log.warn("Refusing to remove variable '{}' — it is a registered graph output", varToRemove);
            return;
        }
        sd.getVariables().remove(varToRemove);
        if (helper != null) {
            helper.updateVariable(varToRemove, null);
        }
    }

    /**
     * @deprecated Use {@link #removeVariable(SameDiff, OptimizationHelper, String)} for better performance
     */
    @Deprecated
    public static void removeVariable(@NonNull SameDiff sd, @NonNull String varToRemove){
        removeVariable(sd, null, varToRemove);
    }

}
