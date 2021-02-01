package org.nd4j.codegen.dsl

import org.apache.commons.io.FileUtils
import org.junit.jupiter.api.Test
import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.AtMost
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Exactly
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.impl.java.JavaPoetGenerator
import java.io.File
import java.nio.charset.StandardCharsets
import kotlin.test.assertTrue

class OpBuilderTest {
    private var testDir = createTempDir()

    @Test
    fun opBuilderTest() {
        val outDir = testDir

        val mathNs = Namespace("math") {
            val config = Config("bla"){
                val a = Input(NUMERIC, "a") { description = "This is A!"}
                Input(NUMERIC, "c") { count = AtLeast(1); description = "This is C!"}
                Input(NUMERIC, "e") { defaultValue = a}
                val b = Arg(NUMERIC, "b") { description = "This is B!"}
                Arg(NUMERIC, "d") { count = AtMost(7); description = "This is D!"}
                Arg(NUMERIC, "f") { defaultValue = 12}

                Constraint("Some constraint"){
                    a.isScalar()
                }
                Constraint("Some different constraint"){
                    b eq 7
                }

                Doc(Language.JAVA, DocScope.ALL){
                    "This is some config documentation"
                }
            }
            Op("add") {
                javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic"

                val x = Input(NUMERIC, "x") { description = "First input to add" }
                Input(NUMERIC,"y") { count = AtLeast(1); description = "Second input to add"; defaultValue = x }
                Arg(INT,"shape") { count = AtLeast(1); description = "shape"; defaultValue = intArrayOf(1,2,3) }


                Output(NUMERIC, "z") { description = "Output (x+y)" }


                Doc(Language.ANY, DocScope.ALL) {
                    """
                    (From AddOp) Add op doc text that will appear everywhere - classes, constructors, op creators
                    """.trimIndent()
                }
                Doc(Language.ANY, DocScope.CLASS_DOC_ONLY) {
                    "Add op doc text that will appear in all class docs (javadoc etc)"
                }
                Doc(Language.ANY, DocScope.CONSTRUCTORS_ONLY) {
                    "Add op doc text for constructors only"
                }

            }

            val baseArithmeticOp = Mixin("BaseArithmeticOp") {
                javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic"

                Input(NUMERIC,"x") { count = Exactly(1); description = "First operand to %OPNAME%" }
                Input(NUMERIC,"y") { count = Exactly(1); description = "Second operand" }


                Output(NUMERIC,"z") { description = "Output" }

                Doc(Language.ANY, DocScope.ALL) {
                    "(From BaseArithmeticOp) op doc text that will appear everywhere - classes, constructors, op creators"
                }
                Doc(Language.ANY, DocScope.CLASS_DOC_ONLY) {
                    "op doc text that will appear in all class docs (javadoc etc)"
                }
                Doc(Language.ANY, DocScope.CONSTRUCTORS_ONLY) {
                    "op doc text for constructors only"
                }

            }

            Op("sub", extends = baseArithmeticOp)

            Op("mul", extends = baseArithmeticOp)

            Op("rsub", extends = baseArithmeticOp) {
                isAbstract = false
                javaPackage = "org.nd4j.some.other.package"
                Doc(Language.ANY, DocScope.CREATORS_ONLY) {
                    "(From rsub) This doc section will appear only in creator method for %OPNAME%"
                }
            }

            Op("div"){
                javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic"

                val x = Input(NUMERIC,"x") { description = "First operand to div" }
                val y = Input(NUMERIC,"y") { description = "Second operand to div" }
                val idx = Arg(INT, "idx") { description = "Some kind of Index" }
                Constraint("Compatible Rank"){
                    x.rank() eq idx
                }

                Constraint("Compatible Shapes"){
                    sameShape(x,y)
                }


                Output(NUMERIC,"z") { description = "Output" }

                Doc(Language.ANY, DocScope.ALL) {
                    "op doc text that will appear everywhere - classes, constructors, op creators"
                }
            }

            Op("zfoo"){
                javaPackage = "bar"
                javaOpClass = "FooBarOp"
                val x = Input(NUMERIC,"x") { description = "First operand to %OPNAME% (%INPUT_TYPE%)" }
                val y = Input(NUMERIC,"y") { description = "Second operand to div" }
                val z = Arg(ENUM, "fooMode"){ description = "Something or other"; possibleValues = listOf("SOME", "value", "Spam", "eGGs")}
                val bla = useConfig(config)

                Constraint("foo bar"){
                    x.sizeAt(7) eq 7 and y.isScalar()
                }

                Doc(Language.ANY, DocScope.ALL) {
                    "op doc text that will appear everywhere - classes, constructors, op creators"
                }

                Signature(x, y, z, bla)
            }
        }

        val generator = JavaPoetGenerator()
        generator.generateNamespaceNd4j(mathNs, null, outDir, "Nd4jMath")
        val exp = File(outDir, "org/nd4j/linalg/factory/ops/Nd4jMath.java")
        assertTrue(exp.isFile)

        println(FileUtils.readFileToString(exp, StandardCharsets.UTF_8))

    }
}