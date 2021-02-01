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
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {}
            }
        }
        assertEquals("foo: Ops must be documented!", thrown.message)
    }


    @Test
    fun opMustBeDocumentedAndNotEmpty() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
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
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Input(DataType.NUMERIC, "y")

                    Signature(x)
                }
            }
        }
        assertEquals("foo: Signature(x) does not cover all parameters! Missing: y", thrown.message)
    }

    @Test
    fun opSignatureMustCoverAllParameters2() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.NUMERIC, "y")

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
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.NUMERIC, "y") {
                    defaultValue = 7
                }

                Signature(x)
            }
        }
    }

    @Test
    fun opSignatureMustTakeEachParameterOnlyOnce() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.NUMERIC, "y")

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
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.NUMERIC, "y") {
                    defaultValue = 7
                }
                val out = Output(DataType.NUMERIC, "out")

                Signature(out, x)
            }
        }
    }

    @Test
    fun opSignatureMustAllowOutputsOnlyOnce() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.NUMERIC, "y") {
                        defaultValue = 7
                    }
                    val out = Output(DataType.NUMERIC, "out")

                    Signature(out, x, out)
                }
            }
        }

        assertEquals("A parameter may not be used twice in a signature!", thrown.message)
    }

    @Test
    fun opSignatureDefaultValueMustHaveCorrectShape() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.INT, "y") {
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
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") {
                    defaultValue = 2
                }

                Signature(x)
            }
        }
    }

    @Test
    fun opSignatureDefaultValueMustHaveCorrectDataType() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.INT, "y") {
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
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val z = Input(DataType.NUMERIC, "z")
                    val y = Arg(DataType.INT, "y") {
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
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.INT, "y") {
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
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = out.shape()
                }

                Signature(out, x)
            }
        }
    }

    @Test
    fun opSignatureDefaultReferenceChain() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val z = Input(DataType.NUMERIC, "z")
                    val u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                    val v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                    val y = Arg(DataType.INT, "y") {
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
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val z = Input(DataType.NUMERIC, "z") { defaultValue = x }
                val u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                val v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                val y = Arg(DataType.INT, "y") {
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
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val z = Input(DataType.NUMERIC, "z") { defaultValue = x }
                val u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                val v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                val y = Arg(DataType.INT, "y") {
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
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") {
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
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x") { defaultValue = null }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun opSignatureShorthandDefaultParams() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val z = Input(DataType.NUMERIC, "z") { defaultValue = x }
                val u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                val v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                val y = Arg(DataType.INT, "y") {
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
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") { count = AtLeast(0); defaultValue = intArrayOf() }
                val z = Arg(DataType.FLOATING_POINT, "z") { count = Range(2, 5); defaultValue = doubleArrayOf(1.0, 2.0, 3.0) }
                val a = Arg(DataType.BOOL, "a") { count = AtLeast(1); defaultValue = booleanArrayOf(true) }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun opSignatureSupportsArrayDefaultsAtLeast() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
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
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
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
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
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
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
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
            val op = Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") { count = AtLeast(0); defaultValue = intArrayOf() }

                AllParamSignature()
                AllDefaultsSignature()
            }

            assertEquals(2, op.signatures.size)
        }
    }

    @Test
    fun opSignatureHasExpectedNumberOfSignaturesWithOutput() {
        Namespace("math") {
            val op = Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") { count = AtLeast(0); defaultValue = intArrayOf() }

                AllParamSignature(true)
                AllDefaultsSignature(true)
            }

            assertEquals(4, op.signatures.size)
        }
    }

    @Test
    fun opSignatureHasExpectedNumberOfSignaturesNoDefaults() {
        Namespace("math") {
            val op = Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") { count = AtLeast(0); }

                AllParamSignature()
                AllDefaultsSignature()
            }

            assertEquals(1, op.signatures.size)
        }
    }

    @Test
    fun opSignatureHasExpectedNumberOfSignaturesWithOutputNoDefaults() {
        Namespace("math") {
            val op = Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") { count = AtLeast(0); }

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
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.ENUM, "y") { possibleValues = listOf("FOO", "BAR", "BAZ"); description = "Enums require some docs" }

            }
        }
    }

    @Test
    fun argEnumDefaultValue() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.ENUM, "y") {
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
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.ENUM, "y") {
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
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.ENUM, "y") {
                        possibleValues = listOf()
                    }

                }
            }
        }

        assertEquals("Arg(ENUM(null), y): Can not set empty possibleValues.", thrown.message)
    }

    @Test
    fun argEnumBadType() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.NUMERIC, "y") {
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
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.ENUM, "y") {
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
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.ENUM, "y") {
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
        val thrown = assertThrows<IllegalArgumentException> {
            val mixin = Mixin("Bar") {
                Input(DataType.NUMERIC, "a")
                Arg(DataType.BOOL, "b")
            }

            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")

                    Signature(out, x, mixin.input("a"), mixin.arg("b"))
                }
            }
        }
        assertEquals("You can only use parameters in a signature that are actually defined in Op(opName=foo, libnd4jOpName=foo, isAbstract=false)! Did you forget to useMixin(...) a mixin?", thrown.message)
    }

    @Test
    fun onlyValidParametersAreUsedInSignaturesGoodCase() {
        val mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            Op("foo", mixin) {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")

                Signature(out, x, mixin.input("a"), mixin.arg("b"))
            }
        }
    }

    @Test
    fun lastMixinDefinitionWins(){
        val mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            val op = Op("foo", mixin) {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                Output(DataType.NUMERIC, "out")
                Input(DataType.NUMERIC, "a") { count=Exactly(1)}
            }

            assertNotSame(mixin.inputs.find { it.name == "a"}, op.inputs.find { it.name == "a"})
        }

    }

    @Test
    fun lastMixinDefinitionWins2(){
        val mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            val op = Op("foo") {
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
        val mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            val op = Op("foo") {
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
        val mixin = Mixin("Bar") {
            javaPackage = "MixinPackage"
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        Namespace("math") {
            val op = Op("foo") {
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
        val mixin = Mixin("Bar") {
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        val op = Mixin("foo") {
            javaPackage = "fooPackage"
            useMixin(mixin)
        }

        assertEquals("fooPackage", op.javaPackage)

    }

    @Test
    fun mixinDoesOnlyOverwritePropertiesIfSetSetCaseOnMixins(){
        val mixin = Mixin("Bar") {
            javaPackage = "MixinPackage"
            Input(DataType.NUMERIC, "a")
            Arg(DataType.BOOL, "b")
        }

        val op = Mixin("foo") {
            javaPackage = "fooPackage"
            useMixin(mixin)
        }

        assertEquals("MixinPackage", op.javaPackage)
    }
}
