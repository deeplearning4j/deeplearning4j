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