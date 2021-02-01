package org.nd4j.codegen.api

import org.nd4j.codegen.api.doc.DocSection

interface OpLike {
    fun name(): String
    fun inputs(): List<Input>
    fun args(): List<Arg>
    fun configs(): List<Config>
    fun outputs(): List<Output>

    fun allParameters(): List<Parameter>

    fun addInput(input: Input)
    fun addArgument(arg: Arg)
    fun addOutput(output: Output)
    fun addDoc(docs: DocSection)
    fun addSignature(signature: Signature)
    fun addConstraint(constraint: Constraint)
    fun addConfig(config: Config)

    fun Config.input(name: String) = inputs.find { it.name == name }!!
    fun Config.arg(name: String) = args.find { it.name == name }!!

    fun Mixin.input(name: String) = inputs.find { it.name == name }!!
    fun Mixin.arg(name: String) = args.find { it.name == name }!!
    fun Mixin.output(name: String) = outputs.find { it.name == name }!!
    fun Mixin.config(name: String) = configs.find { it.name == name }!!
}

data class Op (
        val opName: String,
        var libnd4jOpName: String? = null,
        var javaOpClass: String? = null,
        var isAbstract: Boolean = false,
        var legacy: Boolean = false,
        var argsFirst: Boolean = false,
        var javaPackage: String? = null,
        val inputs: MutableList<Input> = mutableListOf(),
        val outputs: MutableList<Output> = mutableListOf(),
        val args: MutableList<Arg> = mutableListOf(),
        val constraints: MutableList<Constraint> = mutableListOf(),
        val signatures: MutableList<Signature> = mutableListOf(),
        val doc: MutableList<DocSection> = mutableListOf(),
        val configs: MutableList<Config> = mutableListOf()
): OpLike {
    override fun name() = opName
    override fun inputs(): List<Input> = inputs
    override fun args(): List<Arg> = args
    override fun configs(): List<Config> = configs
    override fun outputs(): List<Output> = outputs
    override fun allParameters(): List<Parameter> = inputs + args + configs


    override fun toString(): String {
        return "Op(opName=$opName, libnd4jOpName=$libnd4jOpName, isAbstract=$isAbstract)"
    }

    override fun addInput(input: Input) { inputs.addOrReplace(input) }
    override fun addArgument(arg: Arg) { args.addOrReplace(arg) }
    override fun addOutput(output: Output) { outputs.addOrReplace(output) }
    override fun addDoc(docs: DocSection){ doc.add(docs) }
    override fun addSignature(signature: Signature){ signatures.add(signature) }
    override fun addConstraint(constraint: Constraint){ constraints.add(constraint) }
    override fun addConfig(config: Config) { configs.addOrReplace(config) }

    /**
     * Check that all required properties are set
     */
    fun checkInvariants() {
        if( !isAbstract && (doc.size == 0 || doc.all { it.text.isNullOrBlank() } != false )){
            throw IllegalStateException("$opName: Ops must be documented!")
        }

        signatures.forEach {
            val opParameters = mutableListOf<Parameter>()
            opParameters.addAll(inputs)
            opParameters.addAll(args)

            val notCovered = opParameters.fold(mutableListOf<Parameter>()){acc, parameter ->
                if(!(it.parameters.contains(parameter) || parameter.defaultValueIsApplicable(it.parameters))){
                    acc.add(parameter)
                }
                acc
            }

            if(notCovered.size > 0){
                throw IllegalStateException("$opName: $it does not cover all parameters! Missing: ${notCovered.joinToString(", ") { it.name() }}")
            }
        }

        args.filter { it.type == DataType.ENUM }.forEach {
            if(it.description == null){
                throw IllegalStateException("$opName: Argument ${it.name} is ENUM but has no documentation!")
            }
        }
    }
}

data class Mixin (
        val name: String,

        val inputs: MutableList<Input> = mutableListOf(),
        val outputs: MutableList<Output> = mutableListOf(),
        val args: MutableList<Arg> = mutableListOf(),
        val constraints: MutableList<Constraint> = mutableListOf(),
        val signatures: MutableList<Signature> = mutableListOf(),
        val doc: MutableList<DocSection> = mutableListOf(),
        val configs: MutableList<Config> = mutableListOf()
): OpLike {
    override fun name() = name
    override fun inputs(): List<Input> = inputs
    override fun args(): List<Arg> = args
    override fun configs(): List<Config> = configs
    override fun outputs(): List<Output> = outputs
    override fun allParameters(): List<Parameter> = inputs + args + configs

    override fun toString(): String {
        return "Mixin($name)"
    }

    override fun addInput(input: Input) { inputs.addOrReplace(input) }
    override fun addArgument(arg: Arg) { args.addOrReplace(arg) }
    override fun addOutput(output: Output) { outputs.addOrReplace(output) }
    override fun addDoc(docs: DocSection){ doc.add(docs) }
    override fun addSignature(signature: Signature){ signatures.add(signature) }
    override fun addConstraint(constraint: Constraint){ constraints.add(constraint) }
    override fun addConfig(config: Config) { configs.addOrReplace(config) }


    var legacyWasSet: Boolean = false
    var legacy: Boolean = false
        set(value) {
            field = value
            legacyWasSet = true
        }
    var javaPackageWasSet: Boolean = false
    var javaPackage: String? = null
        set(value) {
            field = value
            javaPackageWasSet = true
        }

    /**
     * Check that all required properties are set
     */
    fun checkInvariants() {
        signatures.forEach {
            val opParameters = mutableListOf<Parameter>()
            opParameters.addAll(inputs)
            opParameters.addAll(args)

            val notCovered = opParameters.fold(mutableListOf<Parameter>()){acc, parameter ->
                if(!(it.parameters.contains(parameter) || parameter.defaultValueIsApplicable(it.parameters))){
                    acc.add(parameter)
                }
                acc
            }

            if(notCovered.size > 0){
                throw IllegalStateException("$this: $it does not cover all parameters! Missing: ${notCovered.joinToString(", ") { it.name() }}")
            }
        }
    }
}

fun <T: Parameter> MutableList<T>.addOrReplaceAll(params: List<T>){
    params.forEach {
        this.addOrReplace(it)
    }
}

fun <T: Parameter> MutableList<T>.addOrReplace(param: T){
    val found = this.find { it.name() == param.name() }
    if(found != null){
        this.replaceAll { if(it.name() == param.name()){ param } else { it } }
    }else{
        this.add(param)
    }
}