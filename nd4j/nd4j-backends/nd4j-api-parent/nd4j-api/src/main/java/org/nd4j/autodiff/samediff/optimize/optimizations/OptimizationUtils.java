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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

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
     */
    public static void removeOp(@NonNull SameDiff sd, OptimizationHelper helper, @NonNull String opToRemove){
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
                v.getInputsForOp().remove(op.getName());
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
