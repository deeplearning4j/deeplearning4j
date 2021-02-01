
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
package org.nd4j.samediff.frameworkimport.process

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.*
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.ir.IROpDef
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule
import org.nd4j.samediff.frameworkimport.rule.tensor.TensorMappingRule
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.lang.IllegalArgumentException

abstract  class AbstractMappingProcess<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(inputFramework: String,
                                                                                   frameworkVersion: String,
                                                                                   inputFrameworkOpName: String,
                                                                                   inputIndexOverrides: Map<Int,Int> = emptyMap(),
                                                                                   opName: String,
                                                                                   opMappingRegistry: OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE>,
                                                                                   tensorMappingRules: List<out TensorMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>,
                                                                                   attributeMappingRules: List<out AttributeMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>):
    MappingProcess<GRAPH_TYPE, OP_DEF_TYPE,
            NODE_TYPE, TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {

    protected val inputFramework = inputFramework
    protected val frameworkVersion = frameworkVersion
    protected val inputFrameworkOpName = inputFrameworkOpName
    protected val opName = opName
    protected val tensorMappingRules = tensorMappingRules
    protected val attributeMappingRules = attributeMappingRules
    protected var opDef: IROpDef<GRAPH_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, DATA_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE>? = null
    protected val opMappingRegistry = opMappingRegistry
    protected val inputIndexOverrides = inputIndexOverrides
    val nd4jOpDescriptors =  OpDescriptorLoaderHolder.nd4jOpDescriptor


    init {

        tensorMappingRules.forEach { tensorMappingRule ->
            tensorMappingRule.initWithMappingProcess(this)
            tensorMappingRule.mappingNamesToPerform().forEach { (nd4jName, inputFrameworkName) ->
                if(!tensorMappingRule.isInputTensorName(inputFrameworkName)) {
                    throw IllegalArgumentException(
                        "Found invalid input tensor named ${inputFrameworkName} for rule ${tensorMappingRule.name()} and mapping process for op ${opName} and input framework name ${inputFrameworkOpName} with definition being  ${
                            nd4jOpDescriptors.findOp(
                                opName
                            )
                        }"
                    )
                }

                if(!tensorMappingRule.isOutputTensorName(nd4jName)) {
                    throw IllegalArgumentException(
                        "Found invalid output tensor named ${nd4jName} for rule ${tensorMappingRule.name()} and mapping process for op ${opName} and input framework name ${inputFrameworkOpName} with definition being ${
                            nd4jOpDescriptors.findOp(
                                opName
                            )
                        }"
                    )
                }

            }
        }

        attributeMappingRules.forEach {
            it.initWithMappingProcess(this)
            attributeMappingRules.forEach { attributeMappingRule ->
                attributeMappingRule.mappingNamesToPerform().forEach { (nd4jName, inputFrameworkName) ->
                    val inputType = attributeMappingRule.attributeValueTypeFor(inputFrameworkName,this)
                    if(!attributeMappingRule.acceptsInputType(inputType)) {
                        throw IllegalArgumentException("Rule ${attributeMappingRule.name()} for framework $inputFramework does not accept input type ${inputType} for attribute name ${inputFrameworkName} and mapping process for op ${opName} and input framework name ${inputFrameworkOpName}")
                    }

                    val outputType = attributeMappingRule.argDescriptorTypesForOutputName(nd4jName,this)
                    if(!attributeMappingRule.outputsType(outputType)) {
                        throw IllegalArgumentException("Rule ${attributeMappingRule.name()} for framework $inputFramework with input framework name $inputFrameworkName and framework op name $inputFrameworkOpName does not accept output type ${outputType} for attribute name ${nd4jName} and mapping process for op ${opName}")
                    }

                }
            }
        }


        opMappingRegistry.registerMappingProcess(
            inputFrameworkOpName = inputFrameworkOpName,
            processToRegister = this
        )


    }

    override fun indexOverrides(): Map<Int, Int> {
        return inputIndexOverrides
    }

    override fun attributeMappingRules(): List<AttributeMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>> {
        return attributeMappingRules
    }

    override fun tensorMappingRules(): List<TensorMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>> {
        return tensorMappingRules
    }

    override fun applyProcessReverse(input: OpNamespace.OpDescriptor): IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        TODO("Not yet implemented")
    }

    override fun inputFrameworkOpName(): String {
        return inputFrameworkOpName
    }

    override fun opName(): String {
        return opName
    }

    override fun frameworkVersion(): String {
        return frameworkVersion
    }

    override fun inputFramework(): String {
        return inputFramework
    }

    override fun applyProcess(mappingCtx: MappingContext<GRAPH_TYPE,
            NODE_TYPE,
            OP_DEF_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE, DATA_TYPE>
    ): Pair<MappingContext<
            GRAPH_TYPE,
            NODE_TYPE,
            OP_DEF_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>, OpNamespace.OpDescriptor> {
        val descriptorBuilder = OpNamespace.OpDescriptor.newBuilder()
        descriptorBuilder.name = opName()
        tensorMappingRules.forEach {
            it.convertInput(mappingCtx).forEach { descriptor -> run {
                descriptorBuilder.addArgDescriptor(descriptor)
                mappingCtx.descriptorsSoFar().add(descriptor)
            }
            }
        }


        attributeMappingRules.forEach {
            it.convertAttributes(mappingCtx).forEach {
                    descriptor ->
                run {
                    descriptorBuilder.addArgDescriptor(descriptor)
                    mappingCtx.descriptorsSoFar().add(descriptor)
                }
            }
        }

        val fullDescriptor = nd4jOpDescriptors.findOp(opName())
        descriptorBuilder.opDeclarationType = fullDescriptor.opDeclarationType

        return Pair(mappingCtx,descriptorBuilder.build())
    }

    override fun serialize(): MapperNamespace.MapperDeclaration {
        val retBuilder = MapperNamespace.MapperDeclaration.newBuilder()
        retBuilder.frameworkName = inputFramework()
        retBuilder.opName = opName()
        retBuilder.inputFrameworkOpName = inputFrameworkOpName()

        indexOverrides().forEach { indexToOverride, replacementIndex ->
            retBuilder.putIndexOverrides(indexToOverride.toLong(),replacementIndex.toLong())
        }

        tensorMappingRules.forEach {
            retBuilder.addRule(it.serialize().toBuilder())
        }

        attributeMappingRules.forEach {
            retBuilder.addRule(it.serialize().toBuilder())
        }

        return retBuilder.build()
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is AbstractMappingProcess<*, *, *, *, *, *, *>) return false

        if (inputFramework != other.inputFramework) return false
        if (frameworkVersion != other.frameworkVersion) return false
        if (inputFrameworkOpName != other.inputFrameworkOpName) return false
        if (opName != other.opName) return false
        if (tensorMappingRules != other.tensorMappingRules) return false
        if (attributeMappingRules != other.attributeMappingRules) return false
        if (opDef != other.opDef) return false
        if (inputIndexOverrides != other.inputIndexOverrides) return false

        return true
    }

    override fun hashCode(): Int {
        var result = inputFramework.hashCode()
        result = 31 * result + frameworkVersion.hashCode()
        result = 31 * result + inputFrameworkOpName.hashCode()
        result = 31 * result + opName.hashCode()
        result = 31 * result + tensorMappingRules.hashCode()
        result = 31 * result + attributeMappingRules.hashCode()
        result = 31 * result + (opDef?.hashCode() ?: 0)
        result = 31 * result + inputIndexOverrides.hashCode()
        return result
    }

    override fun toString(): String {
        return "AbstractMappingProcess(inputFramework='$inputFramework', frameworkVersion='$frameworkVersion', inputFrameworkOpName='$inputFrameworkOpName', opName='$opName', tensorMappingRules=$tensorMappingRules, attributeMappingRules=$attributeMappingRules, opDef=$opDef, inputIndexOverrides=$inputIndexOverrides, nd4jOpDescriptors=$nd4jOpDescriptors)"
    }
}