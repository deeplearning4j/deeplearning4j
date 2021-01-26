package org.nd4j.codegen.impl.python

import org.apache.commons.io.FileUtils
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.NamespaceOps
import org.nd4j.codegen.api.Op
import org.nd4j.codegen.api.doc.DocTokens
import org.nd4j.codegen.api.generator.Generator
import org.nd4j.codegen.api.generator.GeneratorConfig
import org.nd4j.codegen.util.GenUtil
import java.io.File
import java.io.IOException
import java.nio.charset.StandardCharsets

class KotlinExamplePythonGenerator: Generator {
    override fun language() = Language.PYTHON

    @Throws(IOException::class)
    override fun generateNamespaceNd4j(namespace: NamespaceOps?, config: GeneratorConfig?, directory: File?, className: String?) {
        val f = File(directory, GenUtil.ensureFirstIsCap(namespace!!.name) + ".py")
        val content =
        """
            |class ${GenUtil.ensureFirstIsCap(namespace.name)}:
            |${namespace.ops.filterNot { it.isAbstract }.joinToString("\n\n") { generateMethod(it).addIndent(4) }}
        """.trimMargin()
        FileUtils.writeStringToFile(f, content, StandardCharsets.UTF_8)
    }

    fun generateMethod(op: Op): String  =
            """
                |@staticmethod
                |def ${GenUtil.ensureFirstIsNotCap(op.opName)}(${op.inputs.joinToString(", "){it.name}}):
                |${genDocString(op).addIndent(4)}
                |${"# Execution code here".addIndent(4)}
                
            """.trimMargin()

    fun genDocString(op: Op): String {
        //Following roughly: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
        // python docstring / multiline string delimiter is the same as in kotlin, so use this little workaround
        if (op.args.isNotEmpty()) {
            //Args and default args
            throw UnsupportedOperationException("Generating method with args not yet implemented")
        }
        if(op.outputs.size != 1) throw UnsupportedOperationException("Not yet implemented: Python docstring generation for multiple output ops")

        val docStringDelimiter = "\"\"\""
        return """
                |$docStringDelimiter
                |${op.opName} operation
                |
                |${op.inputs.let { """
                |Args: 
                |${it.joinToString("\n") {
               "|    ${it.name} (ndarray): ${DocTokens.processDocText(it.description, op, DocTokens.GenerationType.ND4J)}"
                }}
                |""".trimMargin() }}
                |${op.outputs.let {"""
                |Returns:
                |    ndarray: ${it[0].name} ${it[0].description?.let {"- ${DocTokens.processDocText(it, op, DocTokens.GenerationType.ND4J)}"
                    }}""".trimMargin()
                }}
                |$docStringDelimiter
            """.trimMargin()
    }

    @Throws(IOException::class)
    override fun generateNamespaceSameDiff(namespace: NamespaceOps?, config: GeneratorConfig?, directory: File?, className: String?) {
        throw UnsupportedOperationException("Not yet implemented")
    }

    private fun String.addIndent(size: Int): String = GenUtil.addIndent(this, size)
}

