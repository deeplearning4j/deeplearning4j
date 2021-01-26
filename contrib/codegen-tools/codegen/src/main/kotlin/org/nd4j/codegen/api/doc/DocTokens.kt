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