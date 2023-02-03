package org.eclipse.deeplearning4j.omnihub.api

data class NamespaceModels @JvmOverloads constructor(
    var name: String,
    var models: MutableList<Model> = mutableListOf()) {

}
