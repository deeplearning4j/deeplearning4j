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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX Shrink operation.
 *
 * ONNX Shrink spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shrink
 *
 * Shrink is a thresholding function that maps values to a reduced range:
 * - If x < -lambd: output = x + bias
 * - If x > lambd: output = x - bias
 * - Otherwise: output = 0
 *
 * This is commonly used in sparse coding and wavelet denoising.
 *
 * Inputs:
 * - input: Input tensor
 *
 * Attributes:
 * - bias: The bias value added/subtracted (default: 0.0)
 * - lambd: The threshold value (default: 0.5)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Shrink"], frameworkName = "onnx")
class Shrink : PreImportHook {

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

        // Get input
        val input = sd.getVariable(op.inputsToOp[0])

        // Get attributes
        val bias = (attributes.getOrDefault("bias", 0.0) as Number).toDouble()
        val lambd = (attributes.getOrDefault("lambd", 0.5) as Number).toDouble()
        val inputDt = input.dataType()

        // Shrink function:
        // if x < -lambd: x + bias
        // if x > lambd: x - bias
        // else: 0

        val lambdConst = sd.constant("${opName}_lambd", Nd4j.scalar(inputDt, lambd))
        val negLambdConst = sd.constant("${opName}_negLambd", Nd4j.scalar(inputDt, -lambd))
        val biasConst = sd.constant("${opName}_bias", Nd4j.scalar(inputDt, bias))
        val zero = sd.constant("${opName}_zero", Nd4j.scalar(inputDt, 0.0))

        // Compute masks for each condition
        val ltMask = sd.lt("${opName}_ltMask", input, negLambdConst)  // x < -lambd
        val gtMask = sd.gt("${opName}_gtMask", input, lambdConst)     // x > lambd

        // Convert masks to float
        val ltMaskFloat = ltMask.castTo(input.dataType())
        val gtMaskFloat = gtMask.castTo(input.dataType())

        // Compute values for each branch
        val plusBias = sd.math.add("${opName}_plusBias", input, biasConst)   // x + bias
        val minusBias = sd.math.sub("${opName}_minusBias", input, biasConst) // x - bias

        // Combine: (x + bias) * ltMask + (x - bias) * gtMask + 0 * neither
        val ltPart = sd.math.mul("${opName}_ltPart", plusBias, ltMaskFloat)
        val gtPart = sd.math.mul("${opName}_gtPart", minusBias, gtMaskFloat)
        val output = sd.math.add(outputNames[0], ltPart, gtPart)

        return mapOf(outputNames[0] to listOf(output))
    }
}
