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
package org.nd4j.samediff.frameworkimport

class FrameworkImportConfig {

    val mapLists: MutableMap<String,MutableList<String>> = mutableMapOf()
    val mapFields: MutableMap<String,String> = mutableMapOf()


    fun hasList(key: String): Boolean {
        return mapLists.containsKey(key)
    }

    fun hasValue(key: String): Boolean {
        return mapFields.containsKey(key)
    }

    fun getVal(key: String): String {
        return mapFields[key]!!
    }

    fun getList(key: String): MutableList<String> {
        return mapLists[key]!!
    }

    fun setList(key: String, values: MutableList<String>) {
        mapLists[key] = values
    }

    fun setVal(key:String, value: String) {
        mapFields[key] = value
    }

    fun addValue(key: String,value: String) {
        mapLists[key]!!.add(value)
    }

    fun getValueFromList(key: String,value: Int): String {
        return mapLists[key]!![value]
    }


    fun setValueFromList(key: String,index: Int,value: String) {
        mapLists[key]!![index] = value
    }


}