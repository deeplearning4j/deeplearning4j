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
package org.nd4j.samediff.frameworkimport.onnx.export

import onnx.Onnx
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp

/**
 * Interface for handling complex operations during ONNX export.
 *
 * Some SameDiff operations require special handling when exporting to ONNX,
 * such as:
 * - Weight format conversions (e.g., NHWC to NCHW for convolutions)
 * - Multi-node expansions (e.g., BatchNorm with running statistics)
 * - Attribute conversions that require graph context
 *
 * Implementations of this interface handle these special cases.
 *
 * @author Adam Gibson
 */
interface PostExportHook {

    /**
     * Check if this hook can handle the given operation.
     *
     * @param opName The SameDiff operation name
     * @return true if this hook should handle the operation
     */
    fun canHandle(opName: String): Boolean

    /**
     * Export the operation to ONNX format.
     *
     * @param sd The SameDiff instance containing the operation
     * @param op The SameDiff operation to export
     * @param graphBuilder The ONNX graph builder to add nodes to
     * @param config The export configuration
     * @return List of ONNX nodes created for this operation
     */
    fun export(
        sd: SameDiff,
        op: SameDiffOp,
        graphBuilder: Onnx.GraphProto.Builder,
        config: OnnxExportConfig
    ): List<Onnx.NodeProto>

    /**
     * Get the list of additional initializers needed for this operation.
     *
     * Some operations may need to add constant tensors to the graph
     * (e.g., reshaped weights, scale factors).
     *
     * @param sd The SameDiff instance
     * @param op The operation being exported
     * @return List of tensor protos to add as initializers
     */
    fun getInitializers(
        sd: SameDiff,
        op: SameDiffOp
    ): List<Onnx.TensorProto> = emptyList()

    /**
     * Get the priority of this hook. Higher priority hooks are checked first.
     * Default priority is 0.
     *
     * @return The hook priority
     */
    fun priority(): Int = 0
}

/**
 * Annotation to mark a class as a post-export hook for specific operations.
 * This can be used for automatic discovery and registration.
 *
 * @property opNames The SameDiff operation names this hook handles
 */
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class ExportHookRule(vararg val opNames: String)

/**
 * Registry for post-export hooks.
 */
object PostExportHookRegistry {
    private val hooks = mutableListOf<PostExportHook>()

    /**
     * Register a post-export hook.
     */
    fun register(hook: PostExportHook) {
        hooks.add(hook)
        hooks.sortByDescending { it.priority() }
    }

    /**
     * Find a hook that can handle the given operation.
     *
     * @param opName The SameDiff operation name
     * @return The hook if found, null otherwise
     */
    fun findHook(opName: String): PostExportHook? {
        return hooks.firstOrNull { it.canHandle(opName) }
    }

    /**
     * Get all registered hooks.
     */
    fun getAllHooks(): List<PostExportHook> = hooks.toList()

    /**
     * Clear all registered hooks.
     */
    fun clear() {
        hooks.clear()
    }
}
