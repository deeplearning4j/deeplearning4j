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
 * A port of cast.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/cast.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Gemm"],frameworkName = "onnx")
class Gemm : PreImportHook  {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#gemm
        //this is actually linear, pytorch is capable of exporting a linear operator
        //as Gemm despite it not actually being linear...
        if(op.inputsToOp.size > 2) {
           val lastVar = sd.getVariable(op.inputsToOp[2])
            val len = lastVar.length()
            val transA = attributes.getOrDefault("transA",0L) as Long
            val transB = attributes.getOrDefault("transB",0L) as Long
            val outputVar = sd.nn().linear(outputNames[0],
                sd.getVariable(op.inputsToOp[0])
                ,sd.getVariable(op.inputsToOp[1])
                ,lastVar,transA > 0,transB > 0,false)
            return mapOf(outputVar.name() to listOf(outputVar))

        } else {
            val alpha = attributes.getOrDefault("alpha", 1.0) as Double
            val beta = attributes.getOrDefault("beta",1.0) as Double
            val transA = attributes.getOrDefault("transA",0L) as Long
            val transB = attributes.getOrDefault("transB",0L) as Long

            val outputVar = sd.linalg().matmul(sd.getVariable(op.inputsToOp[0]),
                sd.getVariable(op.inputsToOp[1]),alpha,beta,transA > 0,transB > 0)
            return mapOf(outputVar.name() to listOf(outputVar))
        }
    }


}
