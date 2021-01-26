package org.nd4j.codegen.util

import org.nd4j.codegen.impl.java.JavaPoetGenerator
import org.nd4j.codegen.ops.Bitwise
import org.nd4j.codegen.ops.Random
import java.io.File

fun main() {
    val outDir = File("F:\\dl4j-builds\\deeplearning4j\\nd4j\\nd4j-backends\\nd4j-api-parent\\nd4j-api\\src\\main\\java\\")
    outDir.mkdirs()

    listOf(Bitwise(), Random()).forEach {
        val generator = JavaPoetGenerator()
        generator.generateNamespaceNd4j(it, null, outDir, it.name + ".java")
    }
}