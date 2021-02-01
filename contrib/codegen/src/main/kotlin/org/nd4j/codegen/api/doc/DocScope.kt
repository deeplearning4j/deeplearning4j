package org.nd4j.codegen.api.doc

import org.nd4j.codegen.api.CodeComponent

enum class DocScope {
    ALL, CLASS_DOC_ONLY, CREATORS_ONLY, CONSTRUCTORS_ONLY;

    fun applies(codeComponent: CodeComponent): Boolean {
        return when (this) {
            ALL -> true
            CLASS_DOC_ONLY -> codeComponent === CodeComponent.CLASS_DOC
            CREATORS_ONLY -> codeComponent === CodeComponent.OP_CREATOR
            CONSTRUCTORS_ONLY -> codeComponent === CodeComponent.CONSTRUCTOR
        }
    }
}