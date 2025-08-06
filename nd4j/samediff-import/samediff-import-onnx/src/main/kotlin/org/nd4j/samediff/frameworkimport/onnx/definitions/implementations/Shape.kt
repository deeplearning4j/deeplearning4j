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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of the ONNX Shape operator.
 *
 * The Shape operator takes a tensor as input and outputs an integer tensor
 * containing the shape (dimensions) of the input tensor.
 *
 * ONNX Shape operator reference:
 * https://github.com/onnx/onnx/blob/master/docs/Operators.md#shape
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Shape"], frameworkName = "onnx")
class Shape : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Get the input tensor whose shape we want to extract
        val inputVariable = sd.getVariable(op.inputsToOp[0])

        // The ONNX Shape operator may have optional 'start' and 'end' attributes
        val start = attributes["start"] as? Long ?: 0L
        val end = attributes["end"] as? Long

        val shapeVariable = if (start > 0 || end != null) {
            // Handle slicing case
            val fullShape = inputVariable.shape()

            if (end != null) {
                val sliceBegin = intArrayOf(start.toInt())
                val sliceSize = intArrayOf((end - start).toInt())
                sd.slice(fullShape, sliceBegin, *sliceSize)
            } else {
                val shapeRank = inputVariable.shape?.size ?: 1
                val sliceBegin = intArrayOf(start.toInt())
                val sliceSize = intArrayOf(shapeRank - start.toInt())
                sd.slice(fullShape, sliceBegin, *sliceSize)

            }
        } else {
            // Default case: get the full shape of the input tensor
             inputVariable.shape()
        }

        val ret = shapeVariable.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(ret))


    }
}
