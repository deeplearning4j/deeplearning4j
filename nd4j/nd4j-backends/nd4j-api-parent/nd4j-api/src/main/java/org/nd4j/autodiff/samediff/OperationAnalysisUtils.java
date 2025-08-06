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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.internal.FrameIter;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.VarId;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * Utility class for analyzing operations and their context
 */
@Slf4j
public class OperationAnalysisUtils {
    
    /**
     * Determine the loop role based on operation type
     */
    public static LoopOperationRole determineLoopRole(String opType) {
        if (opType == null) return LoopOperationRole.REGULAR;
        
        switch (opType.toLowerCase()) {
            case "loopcond":
                return LoopOperationRole.CONDITION;
            case "exit":
                return LoopOperationRole.EXIT;
            case "switch":
                return LoopOperationRole.SWITCH;
            case "nextiteration":
                return LoopOperationRole.NEXT_ITERATION;
            case "enter":
                return LoopOperationRole.ENTER;
            case "merge":
                return LoopOperationRole.MERGE;
            default:
                return LoopOperationRole.REGULAR;
        }
    }
    
    /**
     * Check if operation is critical for loop termination
     */
    public static boolean isTerminationCriticalOperation(String opType) {
        LoopOperationRole role = determineLoopRole(opType);
        return role == LoopOperationRole.CONDITION || 
               role == LoopOperationRole.EXIT || 
               role == LoopOperationRole.SWITCH;
    }
    
    /**
     * Create VarId with proper frame and iteration context
     */
    public static VarId createVarId(String varName, FrameIter frameContext, FrameInfo frameInfo) {
        if (frameContext != null) {
            return new VarId(varName, frameContext.getFrame(), frameContext.getIteration(),null);
        } else if (frameInfo != null && frameInfo.targetFrame != null) {
            return new VarId(varName, frameInfo.targetFrame, 0,null);
        } else {
            return new VarId(varName, "OUTER_FRAME", 0,null);
        }
    }
    
    /**
     * Extract value for analysis, handling different SDValue types
     */
    public static Object extractValueForAnalysis(SDValue value) {
        return extractValueForAnalysis(value, 10);
    }
    
    /**
     * Extract value for analysis with specified max elements for arrays
     */
    public static Object extractValueForAnalysis(SDValue value, int maxElements) {
        if (value == null) return null;
        
        switch (value.getSdValueType()) {
            case TENSOR:
                INDArray tensor = value.getTensorValue();
                if (tensor == null) return null;
                
                if (tensor.isScalar()) {
                    return tensor.getDouble(0);
                } else if (tensor.length() <= maxElements) {
                    return createTensorSummary(tensor, true);
                } else {
                    return createTensorSummary(tensor, false);
                }
                
            case LIST:
                return value.getListValue();
                
            default:
                return value.toString();
        }
    }
    
    /**
     * Create a summary representation of a tensor
     */
    public static Map<String, Object> createTensorSummary(INDArray tensor, boolean includeValues) {
        Map<String, Object> summary = new HashMap<>();
        summary.put("shape", Arrays.toString(tensor.shape()));
        summary.put("dataType", tensor.dataType().toString());
        summary.put("length", tensor.length());
        
        if (tensor.isScalar()) {
            summary.put("value", tensor.getDouble(0));
            summary.put("isScalar", true);
        } else {
            summary.put("isScalar", false);
            
            if (includeValues && tensor.length() <= 10) {
                // Show all values for small tensors
                double[] values = tensor.toDoubleVector();
                summary.put("values", Arrays.toString(values));
            } else {
                // Show statistics for large tensors
                try {
                    summary.put("min", tensor.minNumber().doubleValue());
                    summary.put("max", tensor.maxNumber().doubleValue());
                    summary.put("mean", tensor.meanNumber().doubleValue());
                } catch (Exception e) {
                    log.debug("Could not compute tensor statistics: {}", e.getMessage());
                    summary.put("statisticsError", e.getMessage());
                }
            }
            
            // Check for numerical issues
            NumericalHealthInfo healthInfo = analyzeTensorHealth(tensor);
            summary.put("hasNaN", healthInfo.hasNaN);
            summary.put("hasInf", healthInfo.hasInf);
            summary.put("hasExtreme", healthInfo.hasExtreme);
            summary.put("numericalHealth", healthInfo.getHealthDescription());
        }
        
        return summary;
    }
    
    /**
     * Analyze tensor for numerical health issues
     */
    public static NumericalHealthInfo analyzeTensorHealth(INDArray tensor) {
        NumericalHealthInfo healthInfo = new NumericalHealthInfo();
        
        if (tensor == null) {
            healthInfo.isNull = true;
            return healthInfo;
        }
        
        // For large tensors, sample a subset to avoid performance issues
        int sampleSize = Math.min(1000, (int) tensor.length());
        
        for (int i = 0; i < sampleSize; i++) {
            double val = tensor.getDouble(i);
            
            if (Double.isNaN(val)) {
                healthInfo.hasNaN = true;
                healthInfo.nanCount++;
            } else if (Double.isInfinite(val)) {
                healthInfo.hasInf = true;
                healthInfo.infCount++;
            } else if (Math.abs(val) > 1e10) {
                healthInfo.hasExtreme = true;
                healthInfo.extremeCount++;
            }
        }
        
        // If we sampled, extrapolate counts
        if (sampleSize < tensor.length()) {
            double scaleFactor = (double) tensor.length() / sampleSize;
            healthInfo.nanCount = (int) (healthInfo.nanCount * scaleFactor);
            healthInfo.infCount = (int) (healthInfo.infCount * scaleFactor);
            healthInfo.extremeCount = (int) (healthInfo.extremeCount * scaleFactor);
        }
        
        return healthInfo;
    }
    
    /**
     * Estimate memory usage of a value
     */
    public static long estimateValueMemoryUsage(SDValue value) {
        if (value == null) return 0;
        
        switch (value.getSdValueType()) {
            case TENSOR:
                INDArray tensor = value.getTensorValue();
                if (tensor == null) return 0;
                return tensor.length() * tensor.dataType().width();
                
            case LIST:
                // Rough estimate for list values
                Object listValue = value.getListValue();
                if (listValue instanceof List) {
                    return ((List<?>) listValue).size() * 8; // Rough estimate
                }
                return 100;
                
            default:
                return 50; // Placeholder for other types
        }
    }
    
    /**
     * Find the operation that produces a given variable
     */
    public static SameDiffOp findProducerOperation(SameDiff sameDiff, String variableName) {
        for (SameDiffOp op : sameDiff.getOps().values()) {
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains(variableName)) {
                return op;
            }
        }
        return null;
    }
    
    /**
     * Find operations that consume a given variable
     */
    public static List<SameDiffOp> findConsumerOperations(SameDiff sameDiff, String variableName) {
        List<SameDiffOp> consumers = new ArrayList<>();
        
        for (SameDiffOp op : sameDiff.getOps().values()) {
            if (op.getInputsToOp() != null && op.getInputsToOp().contains(variableName)) {
                consumers.add(op);
            }
        }
        
        return consumers;
    }
    
    /**
     * Check if a specific value is problematic (NaN, Inf, etc.)
     */
    public static boolean isProblematicValue(Object value) {
        if (value == null) return false;
        
        if (value instanceof Number) {
            double d = ((Number) value).doubleValue();
            return Double.isNaN(d) || Double.isInfinite(d);
        }
        
        if (value instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) value;
            Boolean hasNaN = (Boolean) map.get("hasNaN");
            Boolean hasInf = (Boolean) map.get("hasInf");
            return (hasNaN != null && hasNaN) || (hasInf != null && hasInf);
        }
        
        return false;
    }
    
    /**
     * Describe the specific problem with a value
     */
    public static String describeProblem(Object value) {
        if (value instanceof Number) {
            double d = ((Number) value).doubleValue();
            if (Double.isNaN(d)) return "NaN value";
            if (Double.isInfinite(d)) return "Infinite value";
            if (Math.abs(d) > 1e10) return "Extreme value: " + d;
        }
        
        if (value instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) value;
            List<String> problems = new ArrayList<>();
            
            Boolean hasNaN = (Boolean) map.get("hasNaN");
            Boolean hasInf = (Boolean) map.get("hasInf");
            Boolean hasExtreme = (Boolean) map.get("hasExtreme");
            
            if (hasNaN != null && hasNaN) problems.add("Contains NaN values");
            if (hasInf != null && hasInf) problems.add("Contains Infinite values");
            if (hasExtreme != null && hasExtreme) problems.add("Contains extreme values");
            
            return problems.isEmpty() ? "Unknown problem" : String.join(", ", problems);
        }
        
        return "Unknown problem";
    }
    
    /**
     * Format value for display
     */
    public static String formatValue(Object value) {
        if (value == null) return "null";
        
        if (value instanceof Number) {
            return String.format("%.6f", ((Number) value).doubleValue());
        }
        
        if (value instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) value;
            
            if (map.containsKey("isScalar") && Boolean.TRUE.equals(map.get("isScalar"))) {
                return String.format("%.6f", (Double) map.get("value"));
            } else {
                StringBuilder sb = new StringBuilder();
                sb.append("Tensor").append(map.get("shape"));
                
                if (map.containsKey("values")) {
                    sb.append(" = ").append(map.get("values"));
                } else {
                    sb.append(" [").append(map.get("dataType")).append("]");
                    if (map.containsKey("mean")) {
                        sb.append(" (mean: ").append(String.format("%.6f", (Double) map.get("mean"))).append(")");
                    }
                }
                
                String health = (String) map.get("numericalHealth");
                if (health != null && !health.equals("HEALTHY")) {
                    sb.append(" [").append(health).append("]");
                }
                
                return sb.toString();
            }
        }
        
        return value.toString();
    }
    
    /**
     * Class to hold numerical health information
     */
    public static class NumericalHealthInfo {
        public boolean isNull = false;
        public boolean hasNaN = false;
        public boolean hasInf = false;
        public boolean hasExtreme = false;
        public int nanCount = 0;
        public int infCount = 0;
        public int extremeCount = 0;
        
        public String getHealthDescription() {
            if (isNull) return "NULL";
            
            List<String> issues = new ArrayList<>();
            if (hasNaN) issues.add(nanCount + " NaN values");
            if (hasInf) issues.add(infCount + " Inf values");
            if (hasExtreme) issues.add(extremeCount + " extreme values");
            
            if (issues.isEmpty()) {
                return "HEALTHY";
            } else {
                return "ISSUES: " + String.join(", ", issues);
            }
        }
        
        public boolean isHealthy() {
            return !isNull && !hasNaN && !hasInf && !hasExtreme;
        }
    }
}
