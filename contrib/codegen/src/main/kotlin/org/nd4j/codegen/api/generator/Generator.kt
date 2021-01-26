package org.nd4j.codegen.api.generator

import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.NamespaceOps
import java.io.File
import java.io.IOException

interface Generator {
    fun language(): Language?

    @Throws(IOException::class)
    fun generateNamespaceNd4j(namespace: NamespaceOps?, config: GeneratorConfig?, directory: File?, className: String?)

    @Throws(IOException::class)
    fun generateNamespaceSameDiff(namespace: NamespaceOps?, config: GeneratorConfig?, directory: File?, className: String?)
}