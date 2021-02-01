package org.nd4j.codegen.api

data class NamespaceOps @JvmOverloads constructor(
    var name: String,
    var include: MutableList<String>? = null,
    var ops: MutableList<Op> = mutableListOf(),
    var configs: MutableList<Config> = mutableListOf(),
    var parentNamespaceOps: Map<String,MutableList<Op>> = mutableMapOf()
) {
    fun addConfig(config: Config) {
        configs.add(config)
    }

    /**
     * Check that all required properties are set
     */
    fun checkInvariants() {
        val usedConfigs = mutableSetOf<Config>()
        ops.forEach { op ->
            usedConfigs.addAll(op.configs)
        }
        val unusedConfigs = configs.toSet() - usedConfigs
        if(unusedConfigs.size > 0){
            throw IllegalStateException("Found unused configs: ${unusedConfigs.joinToString(", ") { it.name }}")
        }
    }

    /**
     * Get op by name
     */
    fun op(name:String):Op {
        val op = ops.find { op -> op.opName.equals(name) } ?: throw java.lang.IllegalStateException("Operation $name not found")
        return op
    }
}