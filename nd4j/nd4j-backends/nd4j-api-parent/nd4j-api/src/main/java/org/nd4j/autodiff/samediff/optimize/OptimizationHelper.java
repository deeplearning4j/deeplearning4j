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


package org.nd4j.autodiff.samediff.optimize;

import lombok.Getter;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.array.OptimizedGraphArrayHolder;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.function.Supplier;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class OptimizationHelper {

    private final SameDiff originalGraph;
    @Getter
    private final Properties properties;
    private boolean setConstantHolder = false;
    private boolean setVariableHolder = false;

    // Fast lookup caches to avoid PatriciaTrie overhead during optimization
    private Map<String, Variable> variablesCache;
    private Map<String, SameDiffOp> opsCache;

    public OptimizationHelper(SameDiff originalGraph, Properties properties){
        this.originalGraph = originalGraph;
        this.properties = properties;
    }

    /**
     * Initialize fast lookup caches from the given SameDiff graph.
     * This converts the PatriciaTrie to a HashMap for O(1) lookups during optimization.
     */
    public void initializeCaches(SameDiff sd) {
        // Create HashMap snapshot of variables for O(1) lookup instead of PatriciaTrie O(k)
        this.variablesCache = new HashMap<>(sd.getVariables());
        this.opsCache = new HashMap<>(sd.getOps());
    }

    /**
     * Fast O(1) variable lookup. Use this instead of sd.getVariables().get() in optimizers.
     */
    public Variable getVariable(String name) {
        return variablesCache != null ? variablesCache.get(name) : null;
    }

    /**
     * Fast O(1) op lookup. Use this instead of sd.getOps().get() in optimizers.
     */
    public SameDiffOp getOp(String name) {
        return opsCache != null ? opsCache.get(name) : null;
    }

    /**
     * Update the variables cache when variables are modified during optimization.
     */
    public void updateVariable(String name, Variable var) {
        if (variablesCache != null) {
            if (var == null) {
                variablesCache.remove(name);
            } else {
                variablesCache.put(name, var);
            }
        }
    }

    /**
     * Update the ops cache when ops are modified during optimization.
     */
    public void updateOp(String name, SameDiffOp op) {
        if (opsCache != null) {
            if (op == null) {
                opsCache.remove(name);
            } else {
                opsCache.put(name, op);
            }
        }
    }

    public OptimizationHelper arrayRecoveryFunction(String arrayName, Supplier<INDArray> fn){
        SDVariable v = originalGraph.getVariable(arrayName);

        // The variable might not exist in the original graph (created during optimization)
        // or might have a different type. In these cases, skip setting the recovery function.
        if (v == null) {
            return this;
        }

        VariableType varType = v.getVariableType();
        if (varType != VariableType.VARIABLE && varType != VariableType.CONSTANT) {
            // Not a variable or constant (e.g., ARRAY or PLACEHOLDER) - skip recovery function
            return this;
        }

        if(varType == VariableType.VARIABLE){
            ArrayHolder h = originalGraph.getVariablesArrays();
            if(!setVariableHolder){
                originalGraph.setVariablesArrays(new OptimizedGraphArrayHolder(h));
                h = originalGraph.getVariablesArrays();
                setVariableHolder = true;
            }
            ((OptimizedGraphArrayHolder)h).setFunction(arrayName, fn);
        } else {
            ArrayHolder h = originalGraph.getConstantArrays();
            if(!setConstantHolder){
                originalGraph.setConstantArrays(new OptimizedGraphArrayHolder(h));
                h = originalGraph.getConstantArrays();
                setConstantHolder = true;
            }
            ((OptimizedGraphArrayHolder)h).setFunction(arrayName, fn);
        }

        return this;
    }

}
