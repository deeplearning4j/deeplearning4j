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
package org.nd4j.samediff.frameworkimport.reflect

import io.github.classgraph.ClassGraph
import org.nd4j.samediff.frameworkimport.hooks.NodePreProcessorHook
import org.nd4j.samediff.frameworkimport.hooks.PostImportHook
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.NodePreProcessor
import org.nd4j.samediff.frameworkimport.hooks.annotations.PostHookRule
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.shade.guava.collect.Table
import org.nd4j.shade.guava.collect.TreeBasedTable
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

object ImportReflectionCache {

    val scannedClasses =   ClassGraph().enableAllInfo().
    disableModuleScanning() // added for GraalVM
     .disableDirScanning() // added for GraalVM
     .disableNestedJarScanning() // added for GraalVM
     .disableRuntimeInvisibleAnnotations() // added for GraalVM
    .addClassLoader(ClassLoader.getSystemClassLoader()) // see
    // https://github.com/oracle/graal/issues/470#issuecomment-401022008
        .scan()

    //all relevant node names relevant for
    val preProcessRuleImplementationsByNode: Table<String,String,MutableList<PreImportHook>> = TreeBasedTable.create()
    val postProcessRuleImplementationsByNode: Table<String,String,MutableList<PostImportHook>> = TreeBasedTable.create()
    //all relevant op names hook should be useful for
    val preProcessRuleImplementationsByOp:  Table<String,String,MutableList<PreImportHook>> = TreeBasedTable.create()
    val postProcessRuleImplementationsByOp: Table<String,String,MutableList<PostImportHook>>  = TreeBasedTable.create()
    val nodePreProcessorRuleImplementationByOp: Table<String,String,MutableList<NodePreProcessorHook<GeneratedMessageV3,
            GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,ProtocolMessageEnum>>>  = TreeBasedTable.create()
    init {
        scannedClasses.getClassesImplementing(PreImportHook::class.java.name).filter { input -> input.hasAnnotation(PreHookRule::class.java.name) }.forEach {
            val instance = Class.forName(it.name).getDeclaredConstructor().newInstance() as PreImportHook
            val rule = it.annotationInfo.first { input -> input.name == PreHookRule::class.java.name }
            val nodeNames = rule.parameterValues["nodeNames"].value as Array<String>
            val frameworkName = rule.parameterValues["frameworkName"].value as String
            nodeNames.forEach { nodeName ->
                if(!preProcessRuleImplementationsByNode.contains(frameworkName,nodeName)) {
                    preProcessRuleImplementationsByNode.put(frameworkName,nodeName,ArrayList())
                }

                preProcessRuleImplementationsByNode.get(frameworkName,nodeName)!!.add(instance)

            }
            val opNames = rule.parameterValues["opNames"].value as Array<String>
            opNames.forEach { opName ->
                if(!preProcessRuleImplementationsByOp.contains(frameworkName,opName)) {
                    preProcessRuleImplementationsByOp.put(frameworkName,opName,ArrayList())
                }

                preProcessRuleImplementationsByOp.get(frameworkName,opName)!!.add(instance)
            }
        }

        scannedClasses.getClassesImplementing(PostImportHook::class.java.name).filter { input -> input.hasAnnotation(PostHookRule::class.java.name) }.forEach {
            val instance = Class.forName(it.name).getDeclaredConstructor().newInstance() as PostImportHook
            val rule = it.annotationInfo.first { input -> input.name == PostHookRule::class.java.name }
            val nodeNames = rule.parameterValues["nodeNames"].value as Array<String>
            val frameworkName = rule.parameterValues["frameworkName"].value as String

            nodeNames.forEach { nodeName ->
                if(!postProcessRuleImplementationsByNode.contains(frameworkName,nodeName)) {
                    postProcessRuleImplementationsByNode.put(frameworkName,nodeName,ArrayList())
                }

                postProcessRuleImplementationsByNode.get(frameworkName,nodeName)!!.add(instance)
            }

            val opNames = rule.parameterValues["opNames"].value as Array<String>
            opNames.forEach { opName ->
                if(!postProcessRuleImplementationsByOp.contains(frameworkName,opName)) {
                    postProcessRuleImplementationsByOp.put(frameworkName,opName,ArrayList())
                }

                postProcessRuleImplementationsByOp.get(frameworkName,opName)!!.add(instance)
            }


        }



        scannedClasses.getClassesImplementing(NodePreProcessorHook::class.java.name).filter { input -> input.hasAnnotation(NodePreProcessor::class.java.name) }.forEach {
            val instance = Class.forName(it.name).getDeclaredConstructor().newInstance() as NodePreProcessorHook<GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,ProtocolMessageEnum>
            val rule = it.annotationInfo.first { input -> input.name == NodePreProcessor::class.java.name }
            val nodeTypes = rule.parameterValues["nodeTypes"].value as Array<String>
            val frameworkName = rule.parameterValues["frameworkName"].value as String
            nodeTypes.forEach { nodeType ->
                if(!nodePreProcessorRuleImplementationByOp.contains(frameworkName,nodeType)) {
                    nodePreProcessorRuleImplementationByOp.put(frameworkName,nodeType,ArrayList())
                }

                nodePreProcessorRuleImplementationByOp.get(frameworkName,nodeType)!!.add(instance)
            }


        }

    }

}

