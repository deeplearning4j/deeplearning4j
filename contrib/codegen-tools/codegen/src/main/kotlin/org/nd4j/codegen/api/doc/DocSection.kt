package org.nd4j.codegen.api.doc

import org.nd4j.codegen.api.CodeComponent
import org.nd4j.codegen.api.Language


data class DocSection(var scope: DocScope? = null,
                 var language: Language? = null,
                 var text: String? = null) {

    fun applies(language: Language, codeComponent: CodeComponent): Boolean {
        return (this.language === Language.ANY || language === this.language) && scope!!.applies(codeComponent)
    }
}