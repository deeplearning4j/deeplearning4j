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
package org.nd4j.samediff.frameworkimport.onnx.export.hooks

import onnx.Onnx
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.samediff.frameworkimport.onnx.*
import org.nd4j.samediff.frameworkimport.onnx.export.ExportHookRule
import org.nd4j.samediff.frameworkimport.onnx.export.OnnxExportConfig
import org.nd4j.samediff.frameworkimport.onnx.export.PostExportHook
import org.slf4j.LoggerFactory

/**
 * Post-export hook for batch normalization operations.
 *
 * Handles:
 * - ONNX BatchNormalization expects inputs: X, scale, B, input_mean, input_var
 * - Epsilon attribute
 * - Momentum attribute
 * - Training mode vs inference mode
 *
 * @author Adam Gibson
 */
@ExportHookRule("batchnorm", "fused_batch_norm")
class BatchNormExportHook : PostExportHook {

    private val logger = LoggerFactory.getLogger(BatchNormExportHook::class.java)

    private val supportedOps = setOf("batchnorm", "fused_batch_norm")

    override fun canHandle(opName: String): Boolean {
        return opName in supportedOps
    }

    override fun export(
        sd: SameDiff,
        op: SameDiffOp,
        graphBuilder: Onnx.GraphProto.Builder,
        config: OnnxExportConfig
    ): List<Onnx.NodeProto> {
        val dynamicOp = op.op as? DynamicCustomOp
            ?: throw IllegalArgumentException("Expected DynamicCustomOp for batchnorm operation")

        val tArgs = dynamicOp.tArgs()
        val bArgs = dynamicOp.bArgs()
        val inputs = op.inputsToOp ?: emptyList()
        val outputs = op.outputsOfOp ?: emptyList()

        // Parse batchnorm arguments
        // tArgs: epsilon
        // bArgs: applyGamma, applyBeta, isTraining
        val epsilon = if (tArgs.isNotEmpty()) tArgs[0] else 1e-5
        val isTraining = if (bArgs.size > 2) bArgs[2] else false

        // Build ONNX BatchNormalization node
        val nodeBuilder = Onnx.NodeProto.newBuilder()
            .setName(op.name)
            .setOpType("BatchNormalization")

        // ONNX BatchNormalization expects inputs in order: X, scale, B, input_mean, input_var
        // SameDiff batchnorm inputs order may differ, need to map correctly
        // Typical SameDiff order: input, mean, variance, gamma, beta
        if (inputs.size >= 5) {
            nodeBuilder.addInput(inputs[0])  // X
            nodeBuilder.addInput(inputs[3])  // scale (gamma)
            nodeBuilder.addInput(inputs[4])  // B (beta)
            nodeBuilder.addInput(inputs[1])  // input_mean
            nodeBuilder.addInput(inputs[2])  // input_var
        } else if (inputs.size >= 3) {
            // Simplified case with fewer inputs
            inputs.forEach { nodeBuilder.addInput(it) }
        } else {
            inputs.forEach { nodeBuilder.addInput(it) }
        }

        // Add outputs
        // ONNX BatchNorm can have up to 5 outputs in training mode:
        // Y, running_mean, running_var, saved_mean, saved_inv_std_dev
        // In inference mode, typically just Y
        if (outputs.isNotEmpty()) {
            nodeBuilder.addOutput(outputs[0])
        }
        // Add additional outputs if present (training mode)
        if (isTraining && outputs.size > 1) {
            for (i in 1 until minOf(outputs.size, 5)) {
                nodeBuilder.addOutput(outputs[i])
            }
        }

        // Add epsilon attribute
        nodeBuilder.addAttribute(AttributeProto {
            name = "epsilon"
            f = epsilon.toFloat()
            type = Onnx.AttributeProto.AttributeType.FLOAT
        })

        // Add training_mode attribute (opset 14+)
        if (config.opsetVersion >= 14) {
            nodeBuilder.addAttribute(AttributeProto {
                name = "training_mode"
                i = if (isTraining) 1 else 0
                type = Onnx.AttributeProto.AttributeType.INT
            })
        }

        logger.debug("Exported batchnorm op: {} with epsilon={}, training={}",
            op.name, epsilon, isTraining)

        return listOf(nodeBuilder.build())
    }

    override fun getInitializers(sd: SameDiff, op: SameDiffOp): List<Onnx.TensorProto> {
        // BatchNorm may need to add default scale/bias if not provided
        return emptyList()
    }

    override fun priority(): Int = 10
}
