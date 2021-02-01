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
package org.nd4j.samediff.frameworkimport.reflect

import io.github.classgraph.ClassGraph
import org.nd4j.samediff.frameworkimport.hooks.PostImportHook
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PostHookRule
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule

object ImportReflectionCache {

    val scannedClasses =   ClassGraph().enableAllInfo()
        .scan()

    //all relevant node names relevant for
    val preProcessRuleImplementationsByNode = HashMap<String,MutableList<PreImportHook>>()
    val postProcessRuleImplementationsByNode = HashMap<String,MutableList<PostImportHook>>()
    //all relevant op names hook should be useful for
    val preProcessRuleImplementationsByOp = HashMap<String,MutableList<PreImportHook>>()
    val postProcessRuleImplementationsByOp = HashMap<String,MutableList<PostImportHook>>()

    init {
        scannedClasses.getClassesImplementing(PreImportHook::class.java.name).filter { input -> input.hasAnnotation(PreHookRule::class.java.name) }.forEach {
            val instance = Class.forName(it.name).getDeclaredConstructor().newInstance() as PreImportHook
            val rule = it.annotationInfo.first { input -> input.name == PreHookRule::class.java.name }
            val nodeNames = rule.parameterValues["nodeNames"].value as Array<String>
            nodeNames.forEach { nodeName ->
                if(!preProcessRuleImplementationsByNode.containsKey(nodeName)) {
                    preProcessRuleImplementationsByNode[nodeName] = ArrayList()
                }

                preProcessRuleImplementationsByNode[nodeName]!!.add(instance)

            }
            val opNames = rule.parameterValues["opNames"].value as Array<String>
            opNames.forEach { opName ->
                if(!preProcessRuleImplementationsByOp.containsKey(opName)) {
                    preProcessRuleImplementationsByNode[opName] = ArrayList()
                }

                preProcessRuleImplementationsByOp[opName]!!.add(instance)
            }
        }

        scannedClasses.getClassesImplementing(PostImportHook::class.java.name).filter { input -> input.hasAnnotation(PostHookRule::class.java.name) }.forEach {
            val instance = Class.forName(it.name).getDeclaredConstructor().newInstance() as PostImportHook
            val rule = it.annotationInfo.first { input -> input.name == PostHookRule::class.java.name }
            val nodeNames = rule.parameterValues["nodeNames"].value as Array<String>
            nodeNames.forEach { nodeName ->
                if(!postProcessRuleImplementationsByNode.containsKey(nodeName)) {
                    postProcessRuleImplementationsByNode[nodeName] = ArrayList()
                }

                postProcessRuleImplementationsByNode[nodeName]!!.add(instance)
            }

            val opNames = rule.parameterValues["opNames"].value as Array<String>
            opNames.forEach { opName ->
                if(!postProcessRuleImplementationsByOp.containsKey(opName)) {
                    postProcessRuleImplementationsByOp[opName] = ArrayList()
                }

                postProcessRuleImplementationsByOp[opName]!!.add(instance)
            }


        }


    }

}

