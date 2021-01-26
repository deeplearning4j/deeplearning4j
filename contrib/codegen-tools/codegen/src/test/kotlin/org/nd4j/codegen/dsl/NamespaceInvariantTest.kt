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