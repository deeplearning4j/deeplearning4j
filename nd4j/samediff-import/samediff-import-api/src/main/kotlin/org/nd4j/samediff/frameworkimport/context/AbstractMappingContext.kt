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
import org.nd4j.samediff.frameworkimport.findOp
import org.nd4j.samediff.frameworkimport.hooks.PostImportHook
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.reflect.ImportReflectionCache
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class AbstractMappingContext<GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        ATTRIBUTE_TYPE: GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum>(
    opDef: OP_DEF_TYPE,
    node: NODE_TYPE,
    graph:
    IRGraph<GRAPH_TYPE,
            NODE_TYPE,
            OP_DEF_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>,
    dynamicVariables: MutableMap<String, TENSOR_TYPE> = HashMap()):
    MappingContext<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {

    val opDef = opDef
    val node = node
    val graph = graph
    val dynamicVariables: MutableMap<String,TENSOR_TYPE> = dynamicVariables
    val descriptorsSoFar = ArrayList<OpNamespace.ArgDescriptor>()
    val relevantPreProcessingHooks = ArrayList<PreImportHook>()
    val relevantPostProcessingHooks = ArrayList<PostImportHook>()
    init {
        discoverHooks()
    }

    fun discoverHooks() {
        ImportReflectionCache.preProcessRuleImplementationsByNode.filterKeys { input -> input == irNode().nodeName() }.values.forEach { hooks ->
            relevantPreProcessingHooks.addAll(hooks)
        }

        ImportReflectionCache.preProcessRuleImplementationsByOp.filterKeys { input -> input == nd4jOpName() }.values.forEach { hooks ->
            relevantPreProcessingHooks.addAll(hooks)
        }


        ImportReflectionCache.postProcessRuleImplementationsByOp.filterKeys { input -> input == nd4jOpName() }.values.forEach { hooks ->
            relevantPostProcessingHooks.addAll(hooks)
        }

        ImportReflectionCache.postProcessRuleImplementationsByNode.filterKeys { input -> input == irNode().nodeName() }.values.forEach { hooks ->
            relevantPostProcessingHooks.addAll(hooks)
        }
    }

    override fun nodeAttributesAsMap(): Map<String, Any> {
        val ret = HashMap<String,Any>()
        irNode().attributeMap().forEach { name, attribute ->
            when(attribute.attributeValueType()) {
                AttributeValueType.LIST_INT -> {
                    ret[name] = attribute.listIntValue()
                }

                AttributeValueType.LIST_TENSOR -> {
                   ret[name] = attribute.listTensorValue().map { input -> input.toNd4jNDArray() }
                }
                AttributeValueType.TENSOR -> {
                    ret[name] = attribute.tensorValue().toNd4jNDArray()
                }

                AttributeValueType.BOOL -> {
                    ret[name] = attribute.boolValue()
                }

                AttributeValueType.DATA_TYPE -> {
                    ret[name] = attribute.dataTataTypeValue().nd4jDataType()
                }

                AttributeValueType.FLOAT -> {
                    ret[name] = attribute.floatValue()
                }

                AttributeValueType.LIST_FLOAT -> {
                    ret[name] = attribute.listFloatValue()
                }

                AttributeValueType.INT -> {
                    ret[name] = attribute.intValue()
                }

                AttributeValueType.LIST_BOOL -> {
                    ret[name] = attribute.listBoolValue()
                }

                AttributeValueType.STRING -> {
                    ret[name] = attribute.stringValue()
                }

                AttributeValueType.LIST_STRING -> {
                    ret[name] = attribute.listStringValue()
                }

                AttributeValueType.INVALID -> {

                }
            }
        }

        return ret
    }

    override fun relevantPrehookRules(): List<PreImportHook> {
        return relevantPreProcessingHooks
    }

    override fun relevantPosthookRules(): List<PostImportHook> {
        return relevantPostProcessingHooks
    }

    override fun descriptorsSoFar(): MutableList<OpNamespace.ArgDescriptor> {
        return descriptorsSoFar
    }

    override fun dynamicResolutionVariables(): MutableMap<String, TENSOR_TYPE> {
        return dynamicVariables
    }

    override fun resolveDynamic(): Boolean {
        return dynamicVariables.isNotEmpty()
    }

    override fun node(): NODE_TYPE {
        return node
    }

    override fun opDef(): OP_DEF_TYPE {
        return opDef
    }

    override fun graph(): IRGraph<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        return graph
    }

    override fun argDescriptorTypeForName(nd4jName: String): List<OpNamespace.ArgDescriptor.ArgType> {
        val opDescriptor = OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(graph.nd4jNameForInternalOpName(opName()))
        return opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.name == nd4jName }.map { argDescriptor ->  argDescriptor.argType }
    }

    override fun nd4jOpName(): String {
        return OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(graph.nd4jNameForInternalOpName(opName())).name
    }
}