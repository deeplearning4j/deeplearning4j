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