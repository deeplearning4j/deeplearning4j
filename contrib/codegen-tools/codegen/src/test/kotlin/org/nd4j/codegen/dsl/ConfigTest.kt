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

import org.junit.jupiter.api.Test
import org.nd4j.codegen.api.DataType.FLOATING_POINT
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope

class ConfigTest {
    @Test
    fun allGood(){
        Namespace("RNN"){
            val sruWeights = Config("SRUWeights"){
                Input(FLOATING_POINT, "weights"){ description = "Weights, with shape [inSize, 3*inSize]" }
                Input(FLOATING_POINT, "bias"){ description = "Biases, with shape [2*inSize]" }
            }

            Op("SRU"){
                Input(FLOATING_POINT, "x"){ description = "..." }
                Input(FLOATING_POINT, "initialC"){ description = "..." }
                Input(FLOATING_POINT, "mask"){ description = "..." }

                useConfig(sruWeights)

                Output(FLOATING_POINT, "out"){ description = "..." }

                Doc(Language.ANY, DocScope.ALL){ "some doc" }
            }

            Op("SRUCell"){
                val x = Input(FLOATING_POINT, "x"){ description = "..." }
                val cLast = Input(FLOATING_POINT, "cLast"){ description = "..." }

                val conf = useConfig(sruWeights)

                Output(FLOATING_POINT, "out"){ description = "..." }

                // Just for demonstration purposes
                Signature(x, cLast, conf)

                Doc(Language.ANY, DocScope.ALL){ "some doc" }
            }
        }
    }
}