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
package org.nd4j.samediff.frameworkimport.onnx.definitions.implementations

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX MaxUnpool operation.
 *
 * ONNX MaxUnpool spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxUnpool
 *
 * MaxUnpool takes the output of MaxPool (values and indices) and computes the
 * partial inverse. Places values back at the positions indicated by indices.
 *
 * Inputs:
 * - X: Input tensor (output values from maxpool)
 * - I: Indices tensor (output indices from maxpool)
 * - output_shape: Optional output shape
 *
 * Attributes:
 * - kernel_shape: Kernel size
 * - pads: Padding values
 * - strides: Stride values
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["MaxUnpool"], frameworkName = "onnx")
class MaxUnpool : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val opName = op.name
        
        // Get inputs
        val input = sd.getVariable(op.inputsToOp[0])  // X: pooled values
        val indices = sd.getVariable(op.inputsToOp[1])  // I: indices from maxpool
        
        // Get optional output shape
        val outputShape = if (op.inputsToOp.size > 2) {
            sd.getVariable(op.inputsToOp[2])
        } else {
            null
        }
        
        // Get attributes
        val kernelShape = getIntList(attributes, "kernel_shape", listOf(2, 2))
        val strides = getIntList(attributes, "strides", listOf(1, 1))
        val pads = getIntList(attributes, "pads", listOf(0, 0, 0, 0))
        
        // Calculate output shape if not provided
        // For NCHW: output_H = (input_H - 1) * stride_H + kernel_H - 2 * pad_H
        val inputShape = sd.shape(input)
        
        val computedShape = if (outputShape != null) {
            outputShape
        } else {
            // Compute based on kernel, stride, and padding
            val kH = kernelShape.getOrElse(0) { 2 }
            val kW = kernelShape.getOrElse(1) { 2 }
            val sH = strides.getOrElse(0) { 1 }
            val sW = strides.getOrElse(1) { 1 }
            val padTop = pads.getOrElse(0) { 0 }
            val padLeft = pads.getOrElse(1) { 0 }
            
            // Get N, C, H, W from input shape
            val n = sd.squeeze(sd.stridedSlice("${opName}_n", inputShape, intArrayOf(0), intArrayOf(1), intArrayOf(1)))
            val c = sd.squeeze(sd.stridedSlice("${opName}_c", inputShape, intArrayOf(1), intArrayOf(2), intArrayOf(1)))
            val h = sd.squeeze(sd.stridedSlice("${opName}_h", inputShape, intArrayOf(2), intArrayOf(3), intArrayOf(1)))
            val w = sd.squeeze(sd.stridedSlice("${opName}_w", inputShape, intArrayOf(3), intArrayOf(4), intArrayOf(1)))
            
            // out_H = (in_H - 1) * stride_H + kernel_H - 2 * pad_top
            val outH = h.sub(1).mul(sH.toDouble()).add(kH.toDouble()).sub(2.0 * padTop)
            val outW = w.sub(1).mul(sW.toDouble()).add(kW.toDouble()).sub(2.0 * padLeft)
            
            sd.stack("${opName}_outshape", 0, n, c, outH, outW).castTo(DataType.INT64)
        }
        
        // Flatten input and indices for scatter operation
        val flatInput = sd.reshape("${opName}_flatInput", input, -1L)
        val flatIndices = sd.reshape("${opName}_flatIndices", indices.castTo(DataType.INT64), -1L)
        
        // Create output tensor of zeros
        val totalSize = sd.prod("${opName}_totalSize", computedShape)
        val zeros = sd.zerosLike(flatInput)
        val zerosOutput = sd.reshape("${opName}_zeros", zeros, totalSize)
        
        // Scatter values into output at indices positions
        // We need to expand indices for scatterNd format
        val indicesExpanded = sd.expandDims("${opName}_indicesExp", flatIndices, -1)
        val scattered = sd.scatterNdUpdate("${opName}_scatter", zerosOutput, indicesExpanded, flatInput)
        
        // Reshape to output shape
        val output = sd.reshape(outputNames[0], scattered, computedShape)
        
        return mapOf(outputNames[0] to listOf(output))
    }
    
    private fun getIntList(attributes: Map<String, Any>, key: String, default: List<Int>): List<Int> {
        val value = attributes[key] ?: return default
        return when (value) {
            is List<*> -> value.map { (it as Number).toInt() }
            is LongArray -> value.map { it.toInt() }
            is IntArray -> value.toList()
            else -> default
        }
    }
}
