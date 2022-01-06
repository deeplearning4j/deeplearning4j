package org.eclipse.deeplearning4j.omnihub.api

data class NamespaceModels @JvmOverloads constructor(
    val name: String,
    val models: MutableList<Model> = mutableListOf()) {

}
