package org.nd4j.codegen.api

object Registry {
    private val enums: MutableMap<String, Arg> = mutableMapOf()
    private val configs: MutableMap<String, Config> = mutableMapOf()

    fun enums() = enums.values.sortedBy { it.name }
    fun configs() = configs.values.sortedBy { it.name }

    fun registerEnum(arg: Arg){
        when(enums[arg.name]){
            null -> enums[arg.name] = arg
            arg -> { /* noop */ }
            else -> throw IllegalStateException("Another enum with the name ${arg.name} already exists! Enums have to use unique names. If you want to use an enum in multiple places, use mixins to define them.")
        }
    }

    fun registerConfig(config: Config){
        when(configs[config.name]){
            null -> configs[config.name] = config
            config -> { /* noop */ }
            else -> throw IllegalStateException("Another config with the name ${config.name} already exists! Configs have to use unique names.")
        }
    }
}