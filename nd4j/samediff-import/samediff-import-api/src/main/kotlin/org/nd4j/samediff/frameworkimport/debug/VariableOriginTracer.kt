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
package org.nd4j.samediff.frameworkimport.debug

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import java.util.concurrent.ConcurrentHashMap

/**
 * Minimal tracer for variable origins during ONNX import debugging.
 * Only active when Environment.isVariableTracingEnabled() is true.
 * 
 * This addresses the core issue described in the technical design document:
 * operations encountering unresolved arrays during import and logging warnings
 * instead of providing actionable debugging information.
 */
object VariableOriginTracer {
    
    private val logger = LoggerFactory.getLogger(VariableOriginTracer::class.java)
    private val origins = ConcurrentHashMap<String, VariableOrigin>()
    
    data class VariableOrigin(
        val varName: String,
        val operation: String,
        val variableType: VariableType?,
        val status: String,  // "resolved", "missing", "placeholder_no_data", "dependency_missing"
        val reason: String,
        val dependencies: Array<String> = emptyArray(),
        val timestamp: Long = System.currentTimeMillis()
    ) {
        override fun toString(): String {
            return "[${status.uppercase()}] $varName in $operation: $reason (${dependencies.size} deps: ${dependencies.contentToString()})"
        }

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as VariableOrigin
            return varName == other.varName &&
                   operation == other.operation &&
                   variableType == other.variableType &&
                   status == other.status &&
                   reason == other.reason &&
                   dependencies.contentEquals(other.dependencies) &&
                   timestamp == other.timestamp
        }

        override fun hashCode(): Int {
            var result = varName.hashCode()
            result = 31 * result + operation.hashCode()
            result = 31 * result + (variableType?.hashCode() ?: 0)
            result = 31 * result + status.hashCode()
            result = 31 * result + reason.hashCode()
            result = 31 * result + dependencies.contentHashCode()
            result = 31 * result + timestamp.hashCode()
            return result
        }
    }
    
    /**
     * Trace a variable resolution attempt during operation execution.
     * This is the core tracing method called from BaseOp and DynamicCustomOp.
     */
    @JvmStatic
    fun traceVariableResolution(varName: String, operation: String, 
                               variable: SDVariable?, array: INDArray?) {
        if (!Nd4j.getEnvironment().isVariableTracingEnabled) {
            return // Zero overhead when disabled
        }
        
        val status: String
        val reason: String
        var dependencies: Array<String>? = null
        val varType = variable?.variableType
        
        when {
            array != null -> {
                status = "resolved"
                reason = "array available, shape: ${array.shape().contentToString()}, dtype: ${array.dataType()}"
            }
            variable?.isPlaceHolder == true -> {
                status = "placeholder_no_data"
                reason = if (variable.shape != null) {
                    "placeholder with shape but no data: ${variable.shape.contentToString()}"
                } else {
                    "placeholder without shape (normal during import)"
                }
            }
            variable != null && varType == VariableType.ARRAY -> {
                status = "dependency_missing"
                var reasonBuilder = "array variable not computed yet, likely dependency chain issue"
                
                // Try to identify what this variable depends on
                variable.sameDiff?.let { sameDiff ->
                    val varMeta = sameDiff.variables[varName]
                    varMeta?.outputOfOp?.let { outputOfOp ->
                        reasonBuilder += " (output of op: $outputOfOp)"
                        // Get dependencies of the operation that should produce this variable
                        val producingOp = sameDiff.ops[outputOfOp]
                        producingOp?.inputsToOp?.let { inputs ->
                            dependencies = inputs.toTypedArray()
                        }
                    }
                }
                reason = reasonBuilder
            }
            else -> {
                status = "missing"
                reason = "no array found and variable type unknown or null"
            }
        }
        
        val origin = VariableOrigin(varName, operation, varType, status, reason, dependencies ?: emptyArray())
        origins["$varName@$operation"] = origin
        
        // Log immediately for problematic cases during import
        when (status) {
            "dependency_missing", "missing" -> logger.warn("VARIABLE_ORIGIN_TRACE: $origin")
            else -> if (logger.isDebugEnabled) logger.debug("VARIABLE_ORIGIN_TRACE: $origin")
        }
    }
    
    /**
     * Convenience method for tracing missing arrays with operation context
     */
    @JvmStatic
    fun traceMissingArray(varName: String, operation: String, reason: String) {
        if (!Nd4j.getEnvironment().isVariableTracingEnabled) {
            return
        }
        
        val origin = VariableOrigin(varName, operation, null, "missing", reason)
        origins["$varName@$operation"] = origin
        logger.warn("VARIABLE_ORIGIN_TRACE: $origin")
    }
    
    /**
     * Get all traced origins
     */
    @JvmStatic
    fun getAllOrigins(): Map<String, VariableOrigin> {
        return HashMap(origins)
    }
    
    /**
     * Clear all traces
     */
    @JvmStatic
    fun clear() {
        origins.clear()
    }
    
    /**
     * Generate a comprehensive report for debugging ONNX import issues
     */
    @JvmStatic
    fun generateReport(): String {
        if (!Nd4j.getEnvironment().isVariableTracingEnabled) {
            return """Variable origin tracing is disabled. Enable with:
                |Nd4j.getEnvironment().setVariableTracingEnabled(true)""".trimMargin()
        }
        
        val sb = StringBuilder()
        sb.append("=== Variable Origin Tracing Report ===\n")
        
        var resolved = 0
        var placeholder = 0
        var dependencyMissing = 0
        var missing = 0
        
        origins.values.forEach { origin ->
            when (origin.status) {
                "resolved" -> resolved++
                "placeholder_no_data" -> placeholder++
                "dependency_missing" -> dependencyMissing++
                "missing" -> missing++
            }
        }
        
        sb.append("Summary: $resolved resolved, $placeholder placeholder, $dependencyMissing dependency_missing, $missing missing\n\n")
        
        if (dependencyMissing > 0) {
            sb.append("Variables with Missing Dependencies (likely cause of import issues):\n")
            origins.values
                .filter { it.status == "dependency_missing" }
                .forEach { sb.append("  $it\n") }
            sb.append("\n")
        }
        
        if (missing > 0) {
            sb.append("Missing Variables:\n")
            origins.values
                .filter { it.status == "missing" }
                .forEach { sb.append("  $it\n") }
            sb.append("\n")
        }
        
        if (placeholder > 0) {
            sb.append("Placeholders without Data (normal during import):\n")
            origins.values
                .filter { it.status == "placeholder_no_data" }
                .forEach { sb.append("  $it\n") }
        }
        
        return sb.toString()
    }
}
