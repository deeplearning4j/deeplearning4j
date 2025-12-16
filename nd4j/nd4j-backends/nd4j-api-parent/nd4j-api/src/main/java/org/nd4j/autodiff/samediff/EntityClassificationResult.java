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
import java.util.*;

/**
 * Entity classification result for execution failure analysis.
 * Contains counts and lists of different entity types that are missing.
 */
@Data
public class EntityClassificationResult {
    
    private final Set<String> operations = new LinkedHashSet<>();
    private final Set<String> variables = new LinkedHashSet<>();
    private final Set<String> constants = new LinkedHashSet<>();
    private final Set<String> placeholders = new LinkedHashSet<>();
    private final Set<String> unknown = new LinkedHashSet<>();
    
    /**
     * Add an entity to the appropriate category
     */
    public void addEntity(String entityName, EntityType type) {
        switch (type) {
            case OPERATION:
                operations.add(entityName);
                break;
            case VARIABLE:
                variables.add(entityName);
                break;
            case CONSTANT:
                constants.add(entityName);
                break;
            case PLACEHOLDER:
                placeholders.add(entityName);
                break;
            case UNKNOWN:
                unknown.add(entityName);
                break;
        }
    }
    
    /**
     * Get total count of all missing entities
     */
    public int getTotalCount() {
        return operations.size() + variables.size() + constants.size() + 
               placeholders.size() + unknown.size();
    }
    
    /**
     * Check if any variables are missing (indicates potential execution framework issue)
     */
    public boolean hasVariables() {
        return !variables.isEmpty();
    }
    
    /**
     * Check if any legitimate operations are missing
     */
    public boolean hasOperations() {
        return !operations.isEmpty();
    }
    
    /**
     * Check if there are any unknown entities
     */
    public boolean hasUnknown() {
        return !unknown.isEmpty();
    }
    
    /**
     * Get count for a specific entity type
     */
    public int getCountForType(EntityType type) {
        switch (type) {
            case OPERATION: return operations.size();
            case VARIABLE: return variables.size();
            case CONSTANT: return constants.size();
            case PLACEHOLDER: return placeholders.size();
            case UNKNOWN: return unknown.size();
            default: return 0;
        }
    }
    
    /**
     * Get entities for a specific type
     */
    public Set<String> getEntitiesForType(EntityType type) {
        switch (type) {
            case OPERATION: return new LinkedHashSet<>(operations);
            case VARIABLE: return new LinkedHashSet<>(variables);
            case CONSTANT: return new LinkedHashSet<>(constants);
            case PLACEHOLDER: return new LinkedHashSet<>(placeholders);
            case UNKNOWN: return new LinkedHashSet<>(unknown);
            default: return new LinkedHashSet<>();
        }
    }
    
    /**
     * Clear all classifications
     */
    public void clear() {
        operations.clear();
        variables.clear();
        constants.clear();
        placeholders.clear();
        unknown.clear();
    }
}
