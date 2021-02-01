/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.samediff.frameworkimport.context

import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.hooks.PostImportHook
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

interface MappingContext<GRAPH_TYPE: GeneratedMessageV3,NODE_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE: GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,DATA_TYPE: ProtocolMessageEnum> {

    /**
     * Prehook rules for this context
     */
    fun relevantPrehookRules(): List<PreImportHook>

    /**
     * Post hook rules for this context
     */
    fun relevantPosthookRules(): List<PostImportHook>

    /**
     * Whether to resolve dynamic place holder variables where
     * scalar values are present. An example scenario is when a value is an input ndarray
     * such as pow(..) where 1 is always an ndarray and the other is a scalar value
     * represented as a double argument in nd4j, but might be a placeholder
     * in the input framework.
     */
    fun resolveDynamic(): Boolean

    /**
     * Return the node attributes as a map.
     * Note: should mainly be used by internal tools
     * that know what to expect from the attributes coming
     * from the context.
     */
    fun nodeAttributesAsMap(): Map<String,Any>

    /**
     * Input variables for  dynamic resolution required for import.
     * This  is important for any cases where  a placeholder variable
     * can be imported and resolved dynamically and later passed on as scalars.
     */
    fun dynamicResolutionVariables(): MutableMap<String, TENSOR_TYPE>

    fun node(): NODE_TYPE

    /**
     * The in use IR Node for mapping
     */
    fun irNode(): IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>



    /**
     * The op def we are mapping
     */
    fun opDef(): OP_DEF_TYPE


    /**
     * The op name we're mapping
     */
    fun opName(): String

    /**
     * The name of the node we're mapping for the context
     */
    fun nodeName(): String

    fun attrDef(name: String): ATTRIBUTE_TYPE

    fun tensorInputFor(name: String): IRTensor<TENSOR_TYPE, DATA_TYPE>


    fun tensorInputFromInputFrameworkName(name: String): IRTensor<TENSOR_TYPE, DATA_TYPE>

    fun tensorAttributeFor(name: String): IRTensor<TENSOR_TYPE, DATA_TYPE>


    fun createIRTensorFromNDArray(ndaray: INDArray): IRTensor<TENSOR_TYPE, DATA_TYPE>

    fun nd4jDataTypeFor(input: IRTensor<TENSOR_TYPE, DATA_TYPE>): DataType

    fun irAttributeValueForNode(valueName: String): IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>

    fun argDescriptorTypeForName(nd4jName: String): List<OpNamespace.ArgDescriptor.ArgType>

    /**
     * Associated graph for the mapping context
     */
    fun graph(): IRGraph<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE, DATA_TYPE>

    /**
     * The op name mapped to in nd4j
     */
    fun nd4jOpName(): String

    /**
     * The descriptors we've accumulated so far for the result
     */
    fun descriptorsSoFar(): MutableList<OpNamespace.ArgDescriptor>
}