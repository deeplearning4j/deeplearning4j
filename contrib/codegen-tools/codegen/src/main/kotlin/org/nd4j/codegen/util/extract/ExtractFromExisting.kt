package org.nd4j.codegen.util.extract

import java.io.File
import java.util.stream.Collectors


private data class Parameter(val name: String, val outType: String)
private data class Op(val name: String, val outType: String, val documentation: String, val parameters: List<Parameter>)

fun main() {
    val inputFile = File("/home/atuzhykov/SkyMind/deeplearning4j/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/ops/SDImage.java")
    val mainRegex = "/\\*\\*(?!\\*)(.*?)\\*/.*?public (.+?) (.+?)\\s*\\((.+?)\\)".toRegex(RegexOption.DOT_MATCHES_ALL)
    val parameterRegex = "(?:@.+?\\s+)?([^\\s,]+?) ([^\\s,]+)".toRegex()

    val contents = inputFile.readText()

    val all = mainRegex.findAll(contents)
    val ops = all.toList().stream().skip(1).map {
        val description = it.groups[1]!!.value.let {
            it.replace("^\\s*\\*".toRegex(RegexOption.MULTILINE), "")
        }
        val outType = it.groups[2]!!.value
        val name = it.groups[3]!!.value
        val parameterString = it.groups[4]!!.value

        val params = parameterRegex.findAll(parameterString).toList().map { Parameter(it.groups[2]!!.value, it.groups[1]!!.value) }


        Op(name, outType, description, params)
    }
            .filter { it.parameters.first().name == "name" }
            .collect(Collectors.toList())


    val out = ops.map { it.toDSL() }.joinToString("\n\n")
    println("""
        import org.nd4j.codegen.api.Language
        import org.nd4j.codegen.api.doc.DocScope
        import org.nd4j.codegen.dsl.*
        import org.nd4j.codegen.api.DataType.*
        
        fun ${inputFile.nameWithoutExtension}() =  Namespace("${inputFile.nameWithoutExtension}"){
            val namespaceJavaPackage = "TODO"
            ${out}
        }
    """.trimIndent())
}

private fun Op.toDSL(): String {
    return """
        Op("${name}") {
            javaPackage = namespaceJavaPackage
            ${parameters.filterNot { it.name == "name" }.map { it.toDSL() }.joinToString("\n            ")}

            Output(NUMERIC, "output"){ description = "" }

            Doc(Language.ANY, DocScope.ALL){
                ${"\"\"\"\n" + documentation + "\n\"\"\".trimIndent()"}
            }
        }
    """.trimIndent()
}

private fun Parameter.toDSL(): String {
    return """Input(NUMERIC, "${name}") { description = "" }"""
}