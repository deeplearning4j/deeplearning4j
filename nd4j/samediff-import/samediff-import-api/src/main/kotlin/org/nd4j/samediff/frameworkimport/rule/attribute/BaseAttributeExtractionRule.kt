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
package org.nd4j.samediff.frameworkimport.rule.attribute

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.findOp
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class BaseAttributeExtractionRule<
        GRAPH_DEF: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(
    name: String,
    mappingNamesToPerform: Map<String, String>,
    transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
    AttributeMappingRule<GRAPH_DEF, OP_DEF_TYPE, NODE_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    protected var opDescriptor: OpNamespace.OpDescriptor? = null
    protected var mappingNamesToPerform = mappingNamesToPerform
    protected var frameworkName: String?  = null
    protected var inputFrameworkOpName: String? = null
    protected var transformerArgs = transformerArgs
    protected var name = name
    protected var inputOpDefTypes: Map<String, AttributeValueType>? = null

    override fun setMappingTransformerArgs(args: Map<String, List<OpNamespace.ArgDescriptor>>) {
        this.transformerArgs = args
    }

    override fun modifyName(name: String) {
        this.name = name
    }

    override fun modifyInputFrameworkOpName(name: String) {
        this.inputFrameworkOpName = name
    }

    override fun initWithMappingProcess(mappingProcess: MappingProcess<GRAPH_DEF, OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>) {
        this.opDescriptor  = OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess.opName())
        this.frameworkName = mappingProcess.inputFramework()
        this.inputFrameworkOpName = mappingProcess.inputFrameworkOpName()
        this.inputOpDefTypes = mappingProcess.inputOpDefValueTypes()
    }

    override fun mappingNamesToPerform(): Map<String, String> {
        return mappingNamesToPerform
    }

    override fun name(): String {
        return name
    }

    override fun mappingTransformerArgs(): Map<String, List<OpNamespace.ArgDescriptor>> {
        return transformerArgs
    }


    abstract fun createIRAttribute(name: String, attrDef: ATTR_DEF, attributeValueType: ATTR_VALUE_TYPE): IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>


    override fun serialize(): MapperNamespace.MappingRule {
        val builder = MapperNamespace.MappingRule.newBuilder()
        builder.ruleName = name()
        builder.functionName = name()
        builder.ruleType = "attribute"
        builder.inputFrameworkOpName = this.inputFrameworkOpName
        val descriptorList = opDescriptor!!.argDescriptorList
        println("Serializing op ${opDescriptor!!.name}")
        for ((k, v) in transformerArgs) {
            v.forEach { descriptor ->
                when (descriptor.argType) {
                    OpNamespace.ArgDescriptor.ArgType.STRING -> builder.addInputStringAttrName(descriptor.name)
                    OpNamespace.ArgDescriptor.ArgType.BOOL -> builder.addInputBooleanName(descriptor.name)
                    OpNamespace.ArgDescriptor.ArgType.DOUBLE, OpNamespace.ArgDescriptor.ArgType.FLOAT -> builder.addInputFloatName(descriptor.name)
                    OpNamespace.ArgDescriptor.ArgType.INT32, OpNamespace.ArgDescriptor.ArgType.INT64 -> builder.addInputIntName(descriptor.name)
                    OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> builder.addInputTensorName(descriptor.name)
                }

                builder.addTransformerArgs(MapperNamespace.TransformerArgs.newBuilder().setKey(k).addAllTransformerArgs(v))
            }

        }

        /**
         * TODO: metadata (perhaps looking up from each framework for each attribute)
         * what each named type is.
         */
        mappingNamesToPerform.forEach { outputName, inputName ->
            val descriptorForName = opDescriptor!!.argDescriptorList.first { descriptor -> descriptor.name == outputName }
            builder.putInputToOutput(outputName,inputName)
            when(descriptorForName.argType) {
                OpNamespace.ArgDescriptor.ArgType.BOOL -> { builder.addOutputBooleanName(outputName)}
                OpNamespace.ArgDescriptor.ArgType.INT64 -> {builder.addOutputIntName(outputName)}
                OpNamespace.ArgDescriptor.ArgType.DOUBLE -> {builder.addOutputDoubleName(outputName)}
                OpNamespace.ArgDescriptor.ArgType.DATA_TYPE -> builder.addOutputDataTypeName(outputName)
                OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR -> builder.addOutputTensorName(outputName)
                OpNamespace.ArgDescriptor.ArgType.STRING -> builder.addOutputStringAttrName(outputName)
            }

            //not all associated outputs will have inputs
            if(inputOpDefTypes!!.containsKey(inputName)) {
                when(inputOpDefTypes!![inputName]!!) {
                    AttributeValueType.FLOAT -> builder.addInputFloatName(inputName)
                    AttributeValueType.INT -> builder.addInputIntName(inputName)
                    AttributeValueType.BOOL -> builder.addInputBooleanName(inputName)
                    AttributeValueType.STRING -> builder.addInputStringAttrName(inputName)
                    AttributeValueType.DATA_TYPE -> builder.addInputDataTypeName(inputName)
                    AttributeValueType.TENSOR -> builder.addInputTensorName(inputName)
                }

            }



        }


        return builder.build()
    }

    override fun argDescriptorTypesForOutputName(
        name: String, mappingProcess:
        MappingProcess<GRAPH_DEF,OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor.ArgType> {
        val nd4jOpDescriptor = OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess.opName())
        val names = nd4jOpDescriptor.argDescriptorList.map { input -> input.name }
        if(!names.contains(name)) {
            throw java.lang.IllegalArgumentException("Unable to find name $name for op $nd4jOpDescriptor.name")
        }

        return nd4jOpDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.name == name }.map { argDescriptor -> argDescriptor.argType}
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BaseAttributeExtractionRule<*, *, *, *, *, *, *>) return false

        if (mappingNamesToPerform != other.mappingNamesToPerform) return false
        if (frameworkName != other.frameworkName) return false
        if (transformerArgs != other.transformerArgs) return false
        if (name != other.name) return false
        if (inputOpDefTypes != other.inputOpDefTypes) return false

        return true
    }

    override fun hashCode(): Int {
        var result = opDescriptor?.hashCode() ?: 0
        result = 31 * result + mappingNamesToPerform.hashCode()
        result = 31 * result + (frameworkName?.hashCode() ?: 0)
        result = 31 * result + (inputFrameworkOpName?.hashCode() ?: 0)
        result = 31 * result + transformerArgs.hashCode()
        result = 31 * result + name.hashCode()
        result = 31 * result + (inputOpDefTypes?.hashCode() ?: 0)
        return result
    }

    override fun toString(): String {
        return "BaseAttributeExtractionRule(mappingNamesToPerform=$mappingNamesToPerform, frameworkName=$frameworkName, inputFrameworkOpName=$inputFrameworkOpName, transformerArgs=$transformerArgs, name='$name', inputOpDefTypes=$inputOpDefTypes)"
    }
}