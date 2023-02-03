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

import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.nd4j.codegen.api.*
import org.nd4j.codegen.api.doc.DocScope
import kotlin.test.assertEquals
import kotlin.test.assertNotSame


class OpInvariantTest {

    @Test
    fun opMustBeDocumented() {
        var thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {}
            }
        }
        assertEquals("foo: Ops must be documented!", thrown.message)
    }


    @Test
    fun opMustBeDocumentedAndNotEmpty() {
        var thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "" }
                }
            }
        }
        assertEquals("foo: Ops must be documented!", thrown.message)
    }

    @Test
    fun opMustBeDocumentedWithDoc() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
            }
        }
    }

    @Test
    fun opSignatureMustCoverAllParameters() {
        var thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Input(DataType.NUMERIC, "y")

                    Signature(x)
                }
            }
        }
        assertEquals("foo: Signature(x) does not cover all parameters! Missing: y", thrown.message)
    }

    @Test
    fun opSignatureMustCoverAllParameters2() {
        var thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.NUMERIC, "y")

                    Signature(x)
                }
            }
        }

        assertEquals("foo: Signature(x) does not cover all parameters! Missing: y", thrown.message)
    }

    @Test
    fun opSignatureMustCoverAllParametersWithoutDefaults() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.NUMERIC, "y") {
                    defaultValue = 7
                }

                Signature(x)
            }
        }
    }

    @Test
    fun opSignatureMustTakeEachParameterOnlyOnce() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.NUMERIC, "y")

                    Signature(x, x, x)
                }
            }
        }

        assertEquals("A parameter may not be used twice in a signature!", thrown.message)
    }

    @Test
    fun opSignatureMustAllowOutputs() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.NUMERIC, "y") {
                    defaultValue = 7
                }
                var out = Output(DataType.NUMERIC, "out")

                Signature(out, x)
            }
        }
    }

    @Test
    fun opSignatureMustAllowOutputsOnlyOnce() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.NUMERIC, "y") {
                        defaultValue = 7
                    }
                    var out = Output(DataType.NUMERIC, "out")

                    Signature(out, x, out)
                }
            }
        }

        assertEquals("A parameter may not be used twice in a signature!", thrown.message)
    }

    @Test
    fun opSignatureDefaultValueMustHaveCorrectShape() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.INT, "y") {
                        defaultValue = x.shape()
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y). Got x.shape() (org.nd4j.codegen.api.TensorShapeValue)", thrown.message)
    }

    @Test
    fun opSignatureDefaultValue() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.INT, "y") {
                    defaultValue = 2
                }

                Signature(x)
            }
        }
    }

    @Test
    fun opSignatureDefaultValueMustHaveCorrectDataType() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.INT, "y") {
                        defaultValue = 1.7
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y). Got 1.7 (java.lang.Double)", thrown.message)
    }


    @Test
    fun opSignatureDefaultInputReference() {
        var thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")
                    var z = Input(DataType.NUMERIC, "z")
                    var y = Arg(DataType.INT, "y") {
                        count = AtLeast(1)
                        defaultValue = z.shape()
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("foo: Signature(x) does not cover all parameters! Missing: z, y", thrown.message)
    }

    @Test
    fun opSignatureDefaultOutputReference() {
        var thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.INT, "y") {
                        count = AtLeast(1)
                        defaultValue = out.shape()
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("foo: Signature(x) does not cover all parameters! Missing: y", thrown.message)
    }

    @Test
    fun opSignatureDefaultWithOutputReference() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = out.shape()
                }

                Signature(out, x)
            }
        }
    }

    @Test
    fun opSignatureDefaultReferenceChain() {
        var thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")
                    var z = Input(DataType.NUMERIC, "z")
                    var u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                    var v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                    var y = Arg(DataType.INT, "y") {
                        count = AtLeast(1)
                        defaultValue = v.shape()
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("foo: Signature(x) does not cover all parameters! Missing: z, u, v, y", thrown.message)
    }

    @Test
    fun opSignatureDefaultReferenceChainWorking() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var z = Input(DataType.NUMERIC, "z") { defaultValue = x }
                var u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                var v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                var y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = v.shape()
                }

                Signature(x)
            }
        }
    }

    @Test
    fun opSignatureShorthandAllParams() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var z = Input(DataType.NUMERIC, "z") { defaultValue = x }
                var u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                var v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                var y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = v.shape()
                }

                AllParamSignature()
            }
        }

    }

    @Test
    fun opSignatureNullDefaults() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = null
                }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun opSignatureNullDefaultsForInputs() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x") { defaultValue = null }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun opSignatureShorthandDefaultParams() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var z = Input(DataType.NUMERIC, "z") { defaultValue = x }
                var u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                var v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                var y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = v.shape()
                }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun opSignatureSupportsArrayDefaults() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.INT, "y") { count = AtLeast(0); defaultValue = intArrayOf() }
                var z = Arg(DataType.FLOATING_POINT, "z") { count = Range(2, 5); defaultValue = doubleArrayOf(1.0, 2.0, 3.0) }
                var a = Arg(DataType.BOOL, "a") { count = AtLeast(1); defaultValue = booleanArrayOf(true) }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun opSignatureSupportsArrayDefaultsAtLeast() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    Output(DataType.NUMERIC, "out")
                    Input(DataType.NUMERIC, "x")
                    Arg(DataType.INT, "y") { count = AtLeast(1); defaultValue = intArrayOf() }
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y){ count = AtLeast(min=1) }. Got [] ([I)", thrown.message)

    }

    @Test
    fun opSignatureSupportsArrayDefaultsAtMost() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    Output(DataType.NUMERIC, "out")
                    Input(DataType.NUMERIC, "x")
                    Arg(DataType.INT, "y") { count = AtMost(1); defaultValue = intArrayOf(1, 2) }
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y){ count = AtMost(max=1) }. Got [1, 2] ([I)", thrown.message)

    }

    @Test
    fun opSignatureSupportsArrayDefaultsRange() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    Output(DataType.NUMERIC, "out")
                    Input(DataType.NUMERIC, "x")
                    Arg(DataType.INT, "y") { count = Range(3, 7); defaultValue = intArrayOf() }
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y){ count = Range(from=3, to=7) }. Got [] ([I)", thrown.message)
    }

    @Test
    fun opSignatureSupportsArrayDefaultsExactly() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    Output(DataType.NUMERIC, "out")
                    Input(DataType.NUMERIC, "x")
                    Arg(DataType.INT, "y") { count = Exactly(7); defaultValue = intArrayOf() }
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y){ count = Exactly(count=7) }. Got [] ([I)", thrown.message)

    }

    @Test
    fun opSignatureHasExpectedNumberOfSignatures() {
        Namespace("math") {
            var op = Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.INT, "y") { count = AtLeast(0); defaultValue = intArrayOf() }

                AllParamSignature()
                AllDefaultsSignature()
            }

            assertEquals(2, op.signatures.size)
        }
    }

    @Test
    fun opSignatureHasExpectedNumberOfSignaturesWithOutput() {
        Namespace("math") {
            var op = Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.INT, "y") { count = AtLeast(0); defaultValue = intArrayOf() }

                AllParamSignature(true)
                AllDefaultsSignature(true)
            }

            assertEquals(4, op.signatures.size)
        }
    }

    @Test
    fun opSignatureHasExpectedNumberOfSignaturesNoDefaults() {
        Namespace("math") {
            var op = Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.INT, "y") { count = AtLeast(0); }

                AllParamSignature()
                AllDefaultsSignature()
            }

            assertEquals(1, op.signatures.size)
        }
    }

    @Test
    fun opSignatureHasExpectedNumberOfSignaturesWithOutputNoDefaults() {
        Namespace("math") {
            var op = Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.INT, "y") { count = AtLeast(0); }

                AllParamSignature(true)
                AllDefaultsSignature(true)
            }

            assertEquals(2, op.signatures.size)
        }
    }

    @Test
    fun argEnum() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.ENUM, "y") { possibleValues = listOf("FOO", "BAR", "BAZ"); description = "Enums require some docs" }

            }
        }
    }

    @Test
    fun argEnumDefaultValue() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.ENUM, "y") {
                    possibleValues = listOf("FOO", "BAR", "BAZ")
                    defaultValue = "BAZ"
                    description = "Enums require some docs"
                }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun argEnumBadDefaultValue() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.ENUM, "y") {
                        possibleValues = listOf("FOO", "BAR", "BAZ")
                        defaultValue = "SPAM"
                    }

                    AllDefaultsSignature()
                }
            }
        }

        assertEquals("Illegal default value for Arg(ENUM(FOO, BAR, BAZ), y). Got SPAM (java.lang.String)", thrown.message)
    }

    @Test
    fun argEnumEmptyPossibleValues() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.ENUM, "y") {
                        possibleValues = listOf()
                    }

                }
            }
        }

        assertEquals("Arg(ENUM(null), y): Can not set empty possibleValues.", thrown.message)
    }

    @Test
    fun argEnumBadType() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.NUMERIC, "y") {
                        possibleValues = listOf("FOO", "BAR", "BAZ")
                        defaultValue = "SPAM"
                    }

                    AllDefaultsSignature()
                }
            }
        }

        assertEquals("Arg(NUMERIC, y): Can not set possibleValues on non ENUM typed Arg.", thrown.message)
    }

    @Test
    fun argEnumBadCount() {
        var thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")
                    var y = Arg(DataType.ENUM, "y") {
                        count = AtLeast(1)
                        possibleValues = listOf("FOO", "BAR", "BAZ")
                    }

                    AllDefaultsSignature()
                }
            }
        }

        assertEquals("Arg(ENUM(null), y): ENUM typed Arg can not be array", thrown.message)
    }

    @Test
    fun argEnumGoodCount() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")
                var y = Arg(DataType.ENUM, "y") {
                    count = Exactly(1)
                    possibleValues = listOf("FOO", "BAR", "BAZ")
                    description = "Enums require some docs"
                }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun onlyValidParametersAreUsedInSignaturesBadCase() {
        var thrown = assertThrows<IllegalArgumentException> {
            var mixin = Mixin("Bar") {
                Input(DataType.NUMERIC, "a")
                Arg(DataType.BOOL, "b")
            }

            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    var out = Output(DataType.NUMERIC, "out")
                    var x = Input(DataType.NUMERIC, "x")

                    Signature(out, x, mixin.input("a"), mixin.arg("b"))
                }
            }
        }
        assertEquals("You can only use parameters in a signature that are actually defined in Op(opName=foo, libnd4jOpName=foo, isAbstract=false)! Did you forget to useMixin(...) a mixin?", thrown.message)
    }

    @Test
    fun onlyValidParametersAreUsedInSignaturesGoodCase() {
        var mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            Op("foo", mixin) {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                var out = Output(DataType.NUMERIC, "out")
                var x = Input(DataType.NUMERIC, "x")

                Signature(out, x, mixin.input("a"), mixin.arg("b"))
            }
        }
    }

    @Test
    fun lastMixinDefinitionWins(){
        var mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            var op = Op("foo", mixin) {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                Output(DataType.NUMERIC, "out")
                Input(DataType.NUMERIC, "a") { count=Exactly(1)}
            }

            assertNotSame(mixin.inputs.find { it.name == "a"}, op.inputs.find { it.name == "a"})
        }

    }

    @Test
    fun lastMixinDefinitionWins2(){
        var mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            var op = Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                Output(DataType.NUMERIC, "out")
                Input(DataType.NUMERIC, "a")
                useMixin(mixin)
            }

            assertSame(mixin.inputs.find { it.name == "a"}, op.inputs.find { it.name == "a"})
        }

    }

    @Test
    fun mixinDoesOnlyOverwritePropertiesIfSetNoneSetCase(){
        var mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            var op = Op("foo") {
                javaPackage = "fooPackage"
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                Output(DataType.NUMERIC, "out")
                Input(DataType.NUMERIC, "a")
                useMixin(mixin)
            }

            assertEquals("fooPackage", op.javaPackage)
        }

    }

    @Test
    fun mixinDoesOnlyOverwritePropertiesIfSetSetCase(){
        var mixin = Mixin("Bar") {
            javaPackage = "MixinPackage"
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            var op = Op("foo") {
                javaPackage = "fooPackage"
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                Output(DataType.NUMERIC, "out")
                Input(DataType.NUMERIC, "a")
                useMixin(mixin)
            }

            assertEquals("MixinPackage", op.javaPackage)
        }

    }

    @Test
    fun mixinDoesOnlyOverwritePropertiesIfSetNoneSetCaseOnMixins(){
        var mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        var op = Mixin("foo") {
            javaPackage = "fooPackage"
            useMixin(mixin)
        }

        assertEquals("fooPackage", op.javaPackage)

    }

    @Test
    fun mixinDoesOnlyOverwritePropertiesIfSetSetCaseOnMixins(){
        var mixin = Mixin("Bar") {
            javaPackage = "MixinPackage"
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        var op = Mixin("foo") {
            javaPackage = "fooPackage"
            useMixin(mixin)
        }

        assertEquals("MixinPackage", op.javaPackage)
    }
}
