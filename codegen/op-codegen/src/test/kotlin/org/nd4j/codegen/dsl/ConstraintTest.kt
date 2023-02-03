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
import org.nd4j.codegen.api.Arg
import org.nd4j.codegen.api.DataType
import org.nd4j.codegen.api.Expression
import org.nd4j.codegen.api.Input
import org.nd4j.codegen.impl.java.JavaConstraintCodeGenerator
import kotlin.test.assertEquals


class ConstraintTest {


    private fun buildConstraint(block: ConstraintBuilder.() -> Expression): Expression {
        return ConstraintBuilder().block()
    }

    @Test
    fun simpleConstraintTest() {
        var expected = "x.rank() == 3"
        var input = Input(name = "x", type = DataType.INT)
        var constraint = buildConstraint { input.rank() eq 3 }
        var out = JavaConstraintCodeGenerator().generateExpression(constraint)
        assertEquals(expected, out)
    }

    @Test
    fun simple2ConstraintTest() {
        var expected = "(x.rank() == 3) && (x.sizeAt(2) >= 7)"
        var input = Input(name = "x", type = DataType.INT)
        var constraint = buildConstraint { (input.rank() eq 3) and (input.sizeAt(2) gte 7) }
        var out = JavaConstraintCodeGenerator().generateExpression(constraint)
        assertEquals(expected, out)
    }

    @Test
    fun simple3ConstraintTest() {
        var expected = "((x.rank() == 3) || (x.sizeAt(2) >= 7)) || (x.sizeAt(4) < 5)"
        var input = Input(name = "x", type = DataType.INT)
        var constraint = buildConstraint { some(
                input.rank() eq 3,
                input.sizeAt(2) gte 7,
                input.sizeAt(4) lt 5
        ) }
        var out = JavaConstraintCodeGenerator().generateExpression(constraint)
        assertEquals(expected, out)
    }

    @Test
    fun complexConstraintTest() {
        var expected = "(x.rank() == 3) == false"
        var input = Input(name = "x", type = DataType.INT)
        var constraint = buildConstraint { not(input.rank() eq 3) }
        var out = JavaConstraintCodeGenerator().generateExpression(constraint)
        assertEquals(expected, out)
    }

    @Test
    fun argConstraintTest() {
        var expected = "(x.rank() == rank) == false"
        var arg = Arg(name = "rank", type = DataType.NUMERIC)
        var input = Input(name = "x", type = DataType.INT)
        var constraint = buildConstraint { not(input.rank() eq arg) }
        var out = JavaConstraintCodeGenerator().generateExpression(constraint)
        assertEquals(expected, out)
    }

    @Test
    fun specificConstraintTest(){
        var expected = "isSameType(x, y, z)"
        var x = Input(name = "x", type = DataType.INT)
        var y = Input(name = "y", type = DataType.INT)
        var z = Input(name = "z", type = DataType.INT)
        var constraint = buildConstraint { sameType(x, y, z) }
        var out = JavaConstraintCodeGenerator().generateExpression(constraint)
        assertEquals(expected, out)
    }
}