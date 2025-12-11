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

import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;

import java.util.List;
import java.util.Set;

/**
 * A properly structured execution node that separates operations from variables.
 * This fixes the core issue where variables and operations were mixed in dependency chains.
 * 
 * Each ExecutionNode represents either:
 * - A forward operation that consumes input variables and produces output variables
 * - A variable initialization (constant, variable, placeholder)
 * - A control flow operation (Switch, Merge, Enter, Exit, etc.)
 * 
 * @author Alex Gibson
 */
@Data
public class ExecutionNode {
    
    private final String operationName;
    private final DifferentialFunction operation;
    private final List<String> inputVariables;      // Variables this op consumes
    private final List<String> outputVariables;     // Variables this op produces
    private final Set<String> dependsOnOperations;  // Operations that must execute before this one
    private final ExecutionNodeType nodeType;
    private final FrameInfo frameInfo;
    
    public enum ExecutionNodeType {
        /** Regular forward operation (MatMul, Add, etc.) */
        STANDARD_OP,
        
        /** Control flow operations (Switch, Merge, Enter, Exit, NextIteration, LoopCond) */
        CONTROL_FLOW_OP,
        
        /** Operations that produce multiple outputs (Split, etc.) */
        MULTI_OUTPUT_OP,
        
        /** Constant/Variable initialization nodes */
        VARIABLE_INIT,
        
        /** Placeholder value assignment nodes */
        PLACEHOLDER_SET
    }
    
    /**
     * Check if this node has any dependencies
     */
    public boolean hasNoDependencies() {
        return dependsOnOperations.isEmpty();
    }
    
    /**
     * Check if this node is ready to execute given the set of completed operations
     */
    public boolean isReadyToExecute(Set<String> completedOperations) {
        return completedOperations.containsAll(dependsOnOperations);
    }
    
    /**
     * Check if this node produces a specific variable
     */
    public boolean producesVariable(String variableName) {
        return outputVariables.contains(variableName);
    }
    
    /**
     * Check if this node consumes a specific variable
     */
    public boolean consumesVariable(String variableName) {
        return inputVariables.contains(variableName);
    }
    
    /**
     * Get a human-readable description of this node
     */
    public String getDescription() {
        StringBuilder sb = new StringBuilder();
        sb.append(nodeType).append(": ").append(operationName);
        
        if (operation != null) {
            sb.append(" (").append(operation.getClass().getSimpleName()).append(")");
        }
        
        if (!inputVariables.isEmpty()) {
            sb.append(" inputs: ").append(inputVariables);
        }
        
        if (!outputVariables.isEmpty()) {
            sb.append(" outputs: ").append(outputVariables);
        }
        
        if (frameInfo != null && !frameInfo.equals(FrameInfo.OUTER_FRAME)) {
            sb.append(" frame: ").append(frameInfo.getFrameName())
              .append(":").append(frameInfo.getIteration());
        }
        
        return sb.toString();
    }
    
    @Override
    public String toString() {
        return getDescription();
    }
}
