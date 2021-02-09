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

package org.nd4j.codegen.dsl

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.nd4j.codegen.api.DataType

class NamespaceInvariantTest {
    @Test
    fun checkForUnusedConfigs(){
        val thrown = assertThrows<IllegalStateException> {
            Namespace("RNN"){
                Config("SRUWeights"){
                    Input(DataType.FLOATING_POINT, "weights"){ description = "Weights, with shape [inSize, 3*inSize]" }
                    Input(DataType.FLOATING_POINT, "bias"){ description = "Biases, with shape [2*inSize]" }
                }
            }
        }
        assertEquals("Found unused configs: SRUWeights", thrown.message)
    }
}