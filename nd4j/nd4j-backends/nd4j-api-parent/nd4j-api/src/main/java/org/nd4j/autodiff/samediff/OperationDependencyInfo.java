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

package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Information about operation dependencies and relationships
 */
@Data
public class OperationDependencyInfo {
    /**
     * Map of input variable names to the operations that produce them
     */
    private Map<String, String> inputDependencies = new HashMap<>();
    
    /**
     * Map of output variable names to the operations that consume them
     */
    private Map<String, List<String>> outputDependencies = new HashMap<>();
    
    /**
     * Operations that must execute before this operation
     */
    private List<String> predecessors = new ArrayList<>();
    
    /**
     * Operations that must execute after this operation
     */
    private List<String> successors = new ArrayList<>();
    
    /**
     * Operations in the same frame/scope
     */
    private List<String> siblings = new ArrayList<>();
    
    /**
     * Control dependencies (operations that must execute before this one for control flow reasons)
     */
    private List<String> controlDependencies = new ArrayList<>();
    
    /**
     * Operations that are part of the same loop
     */
    private List<String> loopPeers = new ArrayList<>();
    
    /**
     * Add an input dependency
     * 
     * @param inputName name of the input variable
     * @param producerOperation name of the operation that produces this input
     */
    public void addInputDependency(String inputName, String producerOperation) {
        inputDependencies.put(inputName, producerOperation);
        if (!predecessors.contains(producerOperation)) {
            predecessors.add(producerOperation);
        }
    }
    
    /**
     * Add an output dependency
     * 
     * @param outputName name of the output variable
     * @param consumerOperation name of the operation that consumes this output
     */
    public void addOutputDependency(String outputName, String consumerOperation) {
        outputDependencies.computeIfAbsent(outputName, k -> new ArrayList<>()).add(consumerOperation);
        if (!successors.contains(consumerOperation)) {
            successors.add(consumerOperation);
        }
    }
    
    /**
     * Add a control dependency
     * 
     * @param controlOperation operation that must execute before this one
     */
    public void addControlDependency(String controlOperation) {
        if (!controlDependencies.contains(controlOperation)) {
            controlDependencies.add(controlOperation);
        }
        if (!predecessors.contains(controlOperation)) {
            predecessors.add(controlOperation);
        }
    }
    
    /**
     * Add a sibling operation (in same frame/scope)
     * 
     * @param siblingOperation name of the sibling operation
     */
    public void addSibling(String siblingOperation) {
        if (!siblings.contains(siblingOperation)) {
            siblings.add(siblingOperation);
        }
    }
    
    /**
     * Add a loop peer operation
     * 
     * @param loopPeerOperation name of the operation in the same loop
     */
    public void addLoopPeer(String loopPeerOperation) {
        if (!loopPeers.contains(loopPeerOperation)) {
            loopPeers.add(loopPeerOperation);
        }
    }
    
    /**
     * Get the producer operation for a specific input
     * 
     * @param inputName name of the input variable
     * @return name of the producer operation, or null if not found
     */
    public String getInputProducer(String inputName) {
        return inputDependencies.get(inputName);
    }
    
    /**
     * Get all consumer operations for a specific output
     * 
     * @param outputName name of the output variable
     * @return list of consumer operation names
     */
    public List<String> getOutputConsumers(String outputName) {
        return outputDependencies.getOrDefault(outputName, new ArrayList<>());
    }
    
    /**
     * Check if this operation has input dependencies
     * 
     * @return true if there are input dependencies
     */
    public boolean hasInputDependencies() {
        return !inputDependencies.isEmpty();
    }
    
    /**
     * Check if this operation has output dependencies
     * 
     * @return true if there are output dependencies
     */
    public boolean hasOutputDependencies() {
        return !outputDependencies.isEmpty();
    }
    
    /**
     * Check if this operation has control dependencies
     * 
     * @return true if there are control dependencies
     */
    public boolean hasControlDependencies() {
        return !controlDependencies.isEmpty();
    }
    
    /**
     * Get total number of dependencies
     * 
     * @return total count of all dependencies
     */
    public int getTotalDependencyCount() {
        return predecessors.size() + successors.size() + controlDependencies.size();
    }
    
    /**
     * Check if operation depends on another operation
     * 
     * @param operationName name of the other operation
     * @return true if this operation depends on the other operation
     */
    public boolean dependsOn(String operationName) {
        return predecessors.contains(operationName) || 
               controlDependencies.contains(operationName) ||
               inputDependencies.containsValue(operationName);
    }
    
    /**
     * Check if another operation depends on this operation
     * 
     * @param operationName name of the other operation
     * @return true if the other operation depends on this operation
     */
    public boolean isDependedOnBy(String operationName) {
        return successors.contains(operationName) ||
               outputDependencies.values().stream().anyMatch(consumers -> consumers.contains(operationName));
    }
    
    /**
     * Clear all dependency information
     */
    public void clearDependencies() {
        inputDependencies.clear();
        outputDependencies.clear();
        predecessors.clear();
        successors.clear();
        siblings.clear();
        controlDependencies.clear();
        loopPeers.clear();
    }
    
    /**
     * Get dependency summary as formatted string
     * 
     * @return formatted dependency summary
     */
    public String getDependencySummary() {
        StringBuilder summary = new StringBuilder();
        summary.append("Dependencies: ");
        summary.append("Inputs: ").append(inputDependencies.size());
        summary.append(", Outputs: ").append(outputDependencies.size());
        summary.append(", Control: ").append(controlDependencies.size());
        summary.append(", Predecessors: ").append(predecessors.size());
        summary.append(", Successors: ").append(successors.size());
        
        if (!loopPeers.isEmpty()) {
            summary.append(", Loop peers: ").append(loopPeers.size());
        }
        
        return summary.toString();
    }
}
