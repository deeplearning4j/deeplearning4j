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

object Registry {
    private val enums: MutableMap<String, Arg> = mutableMapOf()
    private val configs: MutableMap<String, Config> = mutableMapOf()

    fun enums() = enums.values.sortedBy { it.name }
    fun configs() = configs.values.sortedBy { it.name }

    fun registerEnum(arg: Arg){
        when(enums[arg.name]){
            null -> enums[arg.name] = arg
            arg -> { /* noop */ }
            else -> throw IllegalStateException("Another enum with the name ${arg.name} already exists! Enums have to use unique names. If you want to use an enum in multiple places, use mixins to define them.")
        }
    }

    fun registerConfig(config: Config){
        when(configs[config.name]){
            null -> configs[config.name] = config
            config -> { /* noop */ }
            else -> throw IllegalStateException("Another config with the name ${config.name} already exists! Configs have to use unique names.")
        }
    }
}