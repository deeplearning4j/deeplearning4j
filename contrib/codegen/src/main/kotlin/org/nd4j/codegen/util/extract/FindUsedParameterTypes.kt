package org.nd4j.codegen.util.extract

import java.io.File
import kotlin.streams.toList


fun main() {
    val inputDir = File("F:\\dl4j-builds\\deeplearning4j\\nd4j\\nd4j-backends\\nd4j-api-parent\\nd4j-api\\src\\main\\java\\org\\nd4j\\autodiff\\samediff\\ops\\")
    val mainRegex = "/\\*\\*(?!\\*)(.*?)\\*/.*?public (.+?) (.+?)\\s*\\((.+?)\\)".toRegex(RegexOption.DOT_MATCHES_ALL)
    val parameterRegex = "(?:@.+?\\s+)?([^\\s,]+?) ([^\\s,]+)".toRegex()


    val pairs = inputDir.listFiles().filterNot { it.name == "SDValidation.java" }.flatMap { inputFile ->
        val contents = inputFile.readText()

        val all = mainRegex.findAll(contents).toList().stream().skip(1).toList()
        all.flatMap {
            val name = it.groups[3]!!.value
            val parameterString = it.groups[4]!!.value
            parameterRegex.findAll(parameterString).toList().mapNotNull { if(!listOf("name", "names").contains(it.groups[2]!!.value)){name to it.groups[1]!!.value}else{null} }
        }
    }

    val groups = pairs.groupBy { it.second }.mapValues { it.value.count() }.toSortedMap()

    groups.forEach { k, v ->
        println("$k: $v")
    }
}