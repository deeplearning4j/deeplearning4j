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

package org.nd4j.codegen.api.doc

import org.nd4j.codegen.api.Op

object DocTokens {
    enum class GenerationType { SAMEDIFF, ND4J }
    private val OPNAME = "%OPNAME%".toRegex()
    private val LIBND4J_OPNAME = "%LIBND4J_OPNAME%".toRegex()
    private val INPUT_TYPE = "%INPUT_TYPE%".toRegex()

    @JvmStatic fun processDocText(doc: String?, op: Op, type: GenerationType): String {
        return doc
                ?.replace(OPNAME, op.opName)
                ?.replace(LIBND4J_OPNAME, op.libnd4jOpName!!)
                ?.replace(INPUT_TYPE, when(type){
                    GenerationType.SAMEDIFF -> "SDVariable"
                    GenerationType.ND4J -> "INDArray"
                }) ?: ""
    }
}