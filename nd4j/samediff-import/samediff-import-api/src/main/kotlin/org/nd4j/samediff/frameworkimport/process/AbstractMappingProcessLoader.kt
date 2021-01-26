/* ******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.samediff.frameworkimport.process

import io.github.classgraph.ClassGraph
import org.apache.commons.lang3.reflect.TypeUtils
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.rule.MappingRule
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule
import org.nd4j.samediff.frameworkimport.rule.tensor.TensorMappingRule
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.util.concurrent.ConcurrentHashMap

abstract class AbstractMappingProcessLoader<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum>( opMappingRegistry : OpMappingRegistry<GRAPH_TYPE,NODE_DEF_TYPE,OP_DEF_TYPE,TENSOR_TYPE,DATA_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE>):
    MappingProcessLoader<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {
    val attributeRules = HashMap<String,Class<out AttributeMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>>()
    val tensorRules = HashMap<String,Class<out TensorMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>>()
    val opMappingRegistry = opMappingRegistry
    init {
        val scannedClasses =   ClassGraph().enableAllInfo()
            .scan()
        scannedClasses.getClassesImplementing(AttributeMappingRule::class.java.name).filter {
                clazz-> !clazz.isAbstract
                && !clazz.isAnnotation
                && !clazz.isInterface
                && clazz.hasAnnotation(MappingRule::class.java.name)
                && clazz.annotationInfo.first { annotationInfo -> annotationInfo.name == MappingRule::class.java.name }
            .parameterValues["frameworkName"].value.toString() == frameworkName()
        }.forEach { classInfo ->
            val ruleName = classInfo.annotationInfo.first { annotationInfo -> annotationInfo.name ==  MappingRule::class.java.name }.parameterValues["ruleName"].value.toString()
            val type = classInfo.annotationInfo.first { annotationInfo -> annotationInfo.name == MappingRule::class.java.name }.parameterValues["type"].value.toString()
            if(type == "attribute") {
                val clazz =  Class.forName(classInfo.name)
                        as Class<out AttributeMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>

                attributeRules[ruleName] = clazz
            } else if(type == "tensor") {
                val clazz =  Class.forName(classInfo.name)
                        as Class<out TensorMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>

                tensorRules[ruleName] = clazz
            }
        }

        scannedClasses.getClassesImplementing(TensorMappingRule::class.java.name).filter {
                clazz-> !clazz.isAbstract
                && !clazz.isAnnotation
                && !clazz.isInterface
                && clazz.hasAnnotation(MappingRule::class.java.name)
                && clazz.annotationInfo.first { annotationInfo -> annotationInfo.name == MappingRule::class.java.name }
            .parameterValues["frameworkName"].value.toString() == frameworkName()
        }.forEach { classInfo ->
            val ruleName = classInfo.annotationInfo.first { annotationInfo -> annotationInfo.name ==  MappingRule::class.java.name }.parameterValues["ruleName"].value.toString()
            val clazz =  Class.forName(classInfo.name)
                    as Class<out TensorMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>

            tensorRules[ruleName] = clazz
        }

    }

    override fun createProcess(declaration: MapperNamespace.MapperDeclaration): MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        val listOfTensorRules = ArrayList<TensorMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>()
        val listOfAttributeRules = ArrayList<AttributeMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>()
        val dictClass = TypeUtils.parameterize(Map::class.java,String::class.java,String::class.java).rawType as Class<Map<String,String>>
        val transformerArgsClass = TypeUtils.parameterize(Map::class.java,String::class.java,
            TypeUtils.parameterize(List::class.java,
                OpNamespace.ArgDescriptor::class.java)).rawType as Class<Map<String,List<OpNamespace.ArgDescriptor>>>
        declaration.ruleList.forEach { rule ->
            when(rule.ruleType) {
                "tensor" -> {
                    val clazz = tensorRuleRegistry()[rule.ruleName]!!.getConstructor(dictClass,transformerArgsClass)
                    val transformerArgs = ConcurrentHashMap<String,List<OpNamespace.ArgDescriptor>>()
                    rule.transformerArgsList.forEach { arg ->
                        transformerArgs[arg.key] = arg.transformerArgsList
                    }

                    val instance = clazz.newInstance(rule.inputToOutputMap,transformerArgs)
                    listOfTensorRules.add(instance)

                }
                "attribute" -> {
                    val transformerArgs = ConcurrentHashMap<String,List<OpNamespace.ArgDescriptor>>()
                    rule.transformerArgsList.forEach { arg ->
                        transformerArgs[arg.key] = arg.transformerArgsList
                    }

                    val constructor = attributeRuleRegistry()[rule.ruleName]!!.constructors.firstOrNull {
                            constructor -> constructor.parameterCount == 1 || constructor.parameterCount == 2
                    }


                    if(constructor == null) {
                        throw IllegalArgumentException("No constructor found with parameter count < 3! Rule name ${rule.ruleName}")
                    }

                    if(constructor!!.parameterCount == 1) {
                        val instance = constructor!!.newInstance(rule.inputToOutputMap) as AttributeMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
                        instance.setMappingTransformerArgs(transformerArgs)
                        instance.setMappingTransformerArgs(transformerArgs)
                        instance.modifyName(rule.ruleName)
                        instance.modifyInputFrameworkOpName(rule.inputFrameworkOpName)
                        listOfAttributeRules.add(instance)
                    } else if(constructor!!.parameterCount == 2) {
                        val instance = constructor.newInstance(rule.inputToOutputMap,transformerArgs) as AttributeMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>

                        instance.setMappingTransformerArgs(transformerArgs)
                        instance.modifyName(rule.ruleName)
                        instance.modifyInputFrameworkOpName(rule.inputFrameworkOpName)

                        listOfAttributeRules.add(instance)
                    } else {
                        throw IllegalArgumentException("No constructor found with parameter count < 3 for op " + declaration.opName)
                    }




                }
            }
        }

        val indexOverridesConverted = HashMap<Int,Int>()
        declaration.indexOverridesMap.forEach { input, output ->
            indexOverridesConverted[input.toInt()] = output.toInt()
        }

        return instantiateMappingProcess(
            inputFrameworkOpName = declaration.inputFrameworkOpName,
            opName = declaration.opName,
            attributeMappingRules = listOfAttributeRules,
            tensorMappingRules = listOfTensorRules,
            opMappingRegistry = opMappingRegistry,
            indexOverrides = indexOverridesConverted)
    }

    abstract fun instantiateMappingProcess(inputFrameworkOpName: String,opName:String,
                                           attributeMappingRules: List<AttributeMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>,
                                           tensorMappingRules: List<TensorMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>,
                                           opMappingRegistry: OpMappingRegistry<GRAPH_TYPE, NODE_DEF_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE>,
                                           indexOverrides: Map<Int,Int>
    ): MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

    override fun tensorRuleRegistry(): Map<String, Class<TensorMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>> {
        return tensorRules as Map<String, Class<TensorMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>
    }

    override fun attributeRuleRegistry(): Map<String, Class<AttributeMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>> {
        return attributeRules as Map<String, Class<AttributeMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>
    }
}