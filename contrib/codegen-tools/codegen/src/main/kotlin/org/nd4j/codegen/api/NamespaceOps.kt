/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.api

data class NamespaceOps @JvmOverloads constructor(
    var name: String,
    var include: MutableList<String>? = null,
    var ops: MutableList<Op> = mutableListOf(),
    var configs: MutableList<Config> = mutableListOf(),
    var parentNamespaceOps: Map<String,MutableList<Op>> = mutableMapOf()
) {
    fun addConfig(config: Config) {
        configs.add(config)
    }

    /**
     * Check that all required properties are set
     */
    fun checkInvariants() {
        val usedConfigs = mutableSetOf<Config>()
        ops.forEach { op ->
            usedConfigs.addAll(op.configs)
        }
        val unusedConfigs = configs.toSet() - usedConfigs
        if(unusedConfigs.size > 0){
            throw IllegalStateException("Found unused configs: ${unusedConfigs.joinToString(", ") { it.name }}")
        }
    }

    /**
     * Get op by name
     */
    fun op(name:String):Op {
        val op = ops.find { op -> op.opName.equals(name) } ?: throw java.lang.IllegalStateException("Operation $name not found")
        return op
    }
}