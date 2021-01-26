package org.nd4j.codegen.impl.java

import org.nd4j.codegen.api.*
import org.nd4j.codegen.api.generator.ConstraintCodeGenerator

class JavaConstraintCodeGenerator: ConstraintCodeGenerator {
    override fun generateExpression(expression: Expression): String = when(expression) {
        is BooleanExpression -> {
            val left = generateReference(expression.left)
            val right = generateReference(expression.right)
            when(expression.op){
                BooleanOperation.EQ -> "$left == $right"
                BooleanOperation.NEQ -> "$left != $right"
                BooleanOperation.LT -> "$left < $right"
                BooleanOperation.LTE -> "$left <= $right"
                BooleanOperation.GT -> "$left > $right"
                BooleanOperation.GTE -> "$left >= $right"
                BooleanOperation.AND -> "$left && $right"
                BooleanOperation.OR -> "$left || $right"
            }
        }
        is SameTypeExpression -> "isSameType(${expression.inputs.joinToString(", "){ it.name }})"
        is SameShapeExpression -> "isSameShape(${expression.inputs.joinToString(", "){ it.name }})"
        is BroadcastableShapesExpression -> "isBroadcastableShapes(${expression.inputs.joinToString(", "){ it.name }})"
    }

    private fun generateReference(reference: Reference): String = when(reference){
        is NumberReference<*> -> reference.value.toString()
        is BooleanReference -> reference.value.toString()
        is InputShapeReference -> "${reference.input.name}.sizeAt(${reference.idx})"
        is InputRankReference -> "${reference.input.name}.rank()"
        is Arg -> reference.name
        is Expression -> "(${generateExpression(reference)})"
    }
}