package org.nd4j.codegen.api.generator

import org.nd4j.codegen.api.Op

class GeneratorConfig {
    fun acceptOp(op: Op?): Boolean = true
}