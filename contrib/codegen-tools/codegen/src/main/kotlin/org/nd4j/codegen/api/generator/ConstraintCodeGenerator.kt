package org.nd4j.codegen.api.generator

import org.nd4j.codegen.api.Expression


interface ConstraintCodeGenerator {
    fun generateExpression(expression: Expression): String
}