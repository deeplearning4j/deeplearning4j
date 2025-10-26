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

import org.nd4j.autodiff.samediff.internal.VarId;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Utility class for entity classification and analysis in SameDiff execution.
 * Provides methods to classify graph entities and analyze missing entities during execution failures.
 */
public class EntityClassificationUtils {
    
    /**
     * Classify a single entity by name
     */
    public static EntityType classifyEntity(String entityName, SameDiff sameDiff) {
        if (sameDiff.getOps().containsKey(entityName)) {
            return EntityType.OPERATION;
        } else if (sameDiff.getVariables().containsKey(entityName)) {
            return EntityType.VARIABLE;
        } else if (sameDiff.getConstantArrays().hasArray(entityName)) {
            return EntityType.CONSTANT;
        } else if (sameDiff.isPlaceHolder(entityName)) {
            return EntityType.PLACEHOLDER;
        } else {
            return EntityType.UNKNOWN;
        }
    }
    
    /**
     * Classify a collection of entities
     */
    public static EntityClassificationResult classifyEntities(Collection<String> entityNames, SameDiff sameDiff) {
        EntityClassificationResult result = new EntityClassificationResult();
        
        for (String entityName : entityNames) {
            EntityType type = classifyEntity(entityName, sameDiff);
            result.addEntity(entityName, type);
        }
        
        return result;
    }
    
    /**
     * Get display icon for entity type
     */
    public static String getEntityTypeIcon(EntityType type) {
        switch (type) {
            case OPERATION: return "⚙️";
            case VARIABLE: return "📊";
            case CONSTANT: return "📋";
            case PLACEHOLDER: return "🔲";
            case UNKNOWN: return "❓";
            default: return "❓";
        }
    }
    
    /**
     * Get display name for entity type
     */
    public static String getEntityTypeName(EntityType type) {
        switch (type) {
            case OPERATION: return "OPERATION";
            case VARIABLE: return "VARIABLE";
            case CONSTANT: return "CONSTANT";
            case PLACEHOLDER: return "PLACEHOLDER";
            case UNKNOWN: return "UNKNOWN ENTITY";
            default: return "UNKNOWN";
        }
    }
    
    /**
     * Find variable locations across all frames
     */
    public static List<VarId> findVariableLocations(String variableName, 
                                                    Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs) {
        List<VarId> locations = new ArrayList<>();
        
        for (VarId varId : nodeValueOutputs.keySet()) {
            if (varId.getVariable().equals(variableName)) {
                locations.add(varId);
            }
        }
        
        return locations;
    }
    
    /**
     * Check if an entity name represents a variable that should be accessed via VarId rather than executed
     */
    public static boolean isVariableEntity(String entityName, SameDiff sameDiff) {
        return classifyEntity(entityName, sameDiff) == EntityType.VARIABLE;
    }
    
    /**
     * Check if an entity name represents an operation that can be executed
     */
    public static boolean isOperationEntity(String entityName, SameDiff sameDiff) {
        return classifyEntity(entityName, sameDiff) == EntityType.OPERATION;
    }
    
    /**
     * Get all entity names of a specific type from SameDiff
     */
    public static Set<String> getEntitiesByType(EntityType type, SameDiff sameDiff) {
        switch (type) {
            case OPERATION:
                return new LinkedHashSet<>(sameDiff.getOps().keySet());
            case VARIABLE:
                return new LinkedHashSet<>(sameDiff.getVariables().keySet());
            case CONSTANT:
                return new LinkedHashSet<>(sameDiff.getConstantArrays().arrayNames());
            case PLACEHOLDER:
                return new LinkedHashSet<>(sameDiff.variables().stream().filter(input -> input.getVariableType() == VariableType.PLACEHOLDER).map(input -> input.name()).collect(Collectors.toList()));
            default:
                return new LinkedHashSet<>();
        }
    }
}
