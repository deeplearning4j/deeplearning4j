package org.nd4j.codegen.dsl

import org.nd4j.codegen.api.*
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.api.doc.DocSection
import org.nd4j.codegen.ops.SDBaseOps
import java.lang.IllegalStateException

fun Namespace(name: String, block: NamespaceOps.() -> Unit): NamespaceOps {
    val ns = NamespaceOps(name)
    ns.block()

    ns.checkInvariants()
    return ns
}

fun Mixin(name: String, block: Mixin.() -> Unit): Mixin {
    return Mixin(name).apply(block).also {
        it.checkInvariants()
    }
}

fun NamespaceOps.Alias(ns:NamespaceOps, opName:String):Op {
        val op:Op? = ns.op(opName)
        op?.let {
            this.parentNamespaceOps[ns.name]?.add(op)
            this.ops.add(op)
            return op
        }
        throw IllegalStateException("Failed to create alias for op: $opName")
}

fun NamespaceOps.Op(name: String, block: Op.() -> Unit): Op {
    val op = Op(name)
    op.libnd4jOpName = name

    op.block()

    if (!op.isAbstract && op.signatures.isEmpty()) {
        op.AllParamSignature()
        op.AllDefaultsSignature()
    }

    op.checkInvariants()

    this.ops.add(op)
    return op
}

fun NamespaceOps.Op(name: String,
                    extends: Mixin,
                    keepArgs: Boolean = true,
                    keepInputs: Boolean = true,
                    keepOutputs: Boolean = true,
                    keepConstraints: Boolean = true,
                    keepSignatures: Boolean = true,
                    keepDocs: Boolean = true,
                    block: (Op.() -> Unit)? = null): Op {
    return this.Op(name) {
        useMixin(extends, keepArgs = keepArgs, keepInputs = keepInputs, keepOutputs = keepOutputs, keepConstraints = keepConstraints, keepSignatures = keepSignatures, keepDocs = keepDocs)

        if (block != null) {
            this.block()
        }
    }
}


fun OpLike.Input(dataType: DataType, name: String, block: (Input.() -> Unit)? = null): Input {
    val input = Input(name, dataType)
    if (block != null) input.block()

    if (!dataType.isTensorDataType()) {
        throw IllegalArgumentException("Invalid datatype for input \"$name\" of op ${this.name()}: inputs arrays cannot have type $dataType - wrong type, or should be Arg type?");
    }

    this.addInput(input)


    return input
}

fun OpLike.Arg(dataType: DataType, name: String, block: (Arg.() -> Unit)? = null): Arg {
    val input = Arg(name, dataType)
    if (block != null) input.block()

    this.addArgument(input)
    if(dataType == DataType.ENUM){
        Registry.registerEnum(input)
    }
    return input
}

fun OpLike.Output(dataType: DataType, name: String, block: (Output.() -> Unit)? = null): Output {
    val output = Output(name, dataType, false)
    if (block != null) output.block()

    if (!dataType.isTensorDataType()) {
        throw IllegalArgumentException("Invalid datatype for output \"$name\" of op ${this.name()}: output arrays cannot have type $dataType");
    }

    this.addOutput(output)
    return output
}

fun OpLike.Doc(language: Language, scope: DocScope, block: DocSection.() -> String): DocSection {
    val doc = DocSection().apply {
        this.language = language
        this.scope = scope
        text = this.block()
    }
    this.addDoc(doc)
    return doc
}

fun OpLike.AllParamSignature(withOutput: Boolean = false) {
    val allParameters = allParameters()

    this.addSignature(Signature(allParameters))
    if (withOutput) {
        val withOutputParams = mutableListOf<Parameter>().also {
            it.addAll(this.outputs())
            it.addAll(allParameters)
        }
        this.addSignature(Signature(withOutputParams))
    }
}

fun OpLike.AllDefaultsSignature(withOutput: Boolean = false) {
    val allParameters = allParameters()

    if (allParameters.find { it.hasDefaultValue() } != null) {
        val params = allParameters.filterNot { it.hasDefaultValue() }
        this.addSignature(Signature(params))
        if (withOutput) {
            val withOutputParams = mutableListOf<Parameter>().also {
                it.addAll(this.outputs())
                it.addAll(allParameters)
            }
            this.addSignature(Signature(withOutputParams))
        }
    }
}

fun OpLike.Signature(vararg params: Parameter, block: (Signature.() -> String)? = null): Signature {
    if (params.toSet().size < params.size) {
        throw IllegalArgumentException("A parameter may not be used twice in a signature!")
    }
    val paramsAndOutputs = allParameters() + outputs()
    if (!paramsAndOutputs.containsAll(params.toList())) {
        throw IllegalArgumentException("You can only use parameters in a signature that are actually defined in $this! Did you forget to useMixin(...) a mixin?")
    }

    val signature = Signature(params.toList())
    if (block != null) {
        signature.block()
    }
    this.addSignature(signature)
    return signature
}

fun OpLike.Constraint(desc: String, block: ConstraintBuilder.() -> Expression): Constraint {
    val check = ConstraintBuilder().block()
    val constraint = Constraint(desc, check)
    this.addConstraint(constraint)
    return constraint
}

fun OpLike.BackendConstraint(desc: String, block: ConstraintBuilder.() -> Expression): Constraint {
    val check = ConstraintBuilder().block()
    val constraint = BackendConstraint(desc, check)
    this.addConstraint(constraint)
    return constraint
}

class ConstraintBuilder {
    fun broadcastableShapes(vararg inputs: Input) = BroadcastableShapesExpression(inputs.toList())
    fun sameShape(vararg inputs: Input) = SameShapeExpression(inputs.toList())
    fun sameType(vararg inputs: Input) = SameTypeExpression(inputs.toList())

    fun Input.sizeAt(i: Int) = InputShapeReference(this, i)
    fun Input.rank() = InputRankReference(this)
    fun Input.isScalar() = this.rank() eq 0

    fun some(expr: BooleanExpression, vararg exprs: BooleanExpression) = exprs.fold(expr, { acc, cur -> acc or cur })
    fun all(expr: BooleanExpression, vararg exprs: BooleanExpression) = exprs.fold(expr, { acc, cur -> acc and cur })
    fun not(expr: BooleanExpression) = expr eq false

    infix fun BooleanExpression.and(other: BooleanExpression) = BooleanExpression(this, other, BooleanOperation.AND)
    infix fun BooleanExpression.or(other: BooleanExpression) = BooleanExpression(this, other, BooleanOperation.OR)


    infix fun Reference.eq(other: Reference) = BooleanExpression(this, other, BooleanOperation.EQ)
    infix fun Reference.eq(other: Number) = this eq NumberReference(other)
    infix fun Reference.eq(other: Boolean) = this eq BooleanReference(other)


    infix fun Reference.neq(other: Reference) = BooleanExpression(this, other, BooleanOperation.NEQ)
    infix fun <T : Number> Reference.neq(other: T) = this neq NumberReference(other)
    infix fun Reference.neq(other: Boolean) = this neq BooleanReference(other)

    infix fun Reference.gt(other: Reference) = BooleanExpression(this, other, BooleanOperation.GT)
    infix fun <T : Number> Reference.gt(other: T) = this gt NumberReference(other)

    infix fun Reference.lt(other: Reference) = BooleanExpression(this, other, BooleanOperation.LT)
    infix fun <T : Number> Reference.lt(other: T) = this lt NumberReference(other)


    infix fun <T : Number> Reference.gte(other: T) = this gte NumberReference(other)
    infix fun Reference.gte(other: Reference) = BooleanExpression(this, other, BooleanOperation.GTE)

    infix fun <T : Number> Reference.lte(other: T) = this lte NumberReference(other)
    infix fun Reference.lte(other: Reference) = BooleanExpression(this, other, BooleanOperation.LTE)
}

fun NamespaceOps.Config(name: String, block: (Config.() -> Unit)): Config {
    val config = Config(name)
    config.block()
    this.addConfig(config)
    Registry.registerConfig(config)
    return config
}

fun Config.Input(dataType: DataType, name: String, block: (Input.() -> Unit)? = null): Input {
    val input = Input(name, dataType)
    if (block != null) input.block()

    if (!dataType.isTensorDataType()) {
        throw IllegalArgumentException("Invalid datatype for input \"$name\" of config ${this.name}: inputs arrays cannot have type $dataType - wrong type, or should be Arg type?");
    }

    this.addInput(input)
    return input
}

fun Config.Arg(dataType: DataType, name: String, block: (Arg.() -> Unit)? = null): Arg {
    val input = Arg(name, dataType)
    if (block != null) input.block()

    this.addArgument(input)
    if(dataType == DataType.ENUM){
        Registry.registerEnum(input)
    }

    return input
}

fun Config.Constraint(desc: String, block: ConstraintBuilder.() -> Expression): Constraint {
    val check = ConstraintBuilder().block()
    val constraint = Constraint(desc, check)
    this.addConstraint(constraint)
    return constraint
}

fun Config.BackendConstraint(desc: String, block: ConstraintBuilder.() -> Expression): Constraint {
    val check = ConstraintBuilder().block()
    val constraint = BackendConstraint(desc, check)
    this.addConstraint(constraint)
    return constraint
}

fun Config.Doc(language: Language, scope: DocScope, block: DocSection.() -> String): DocSection {
    val doc = DocSection().apply {
        this.language = language
        this.scope = scope
        text = this.block()
    }
    this.addDoc(doc)
    return doc
}

fun OpLike.useConfig(config: Config): Config {
    this.addConfig(config)
    return config
}

fun Op.useMixin(mixin: Mixin,
                keepArgs: Boolean = true,
                keepInputs: Boolean = true,
                keepOutputs: Boolean = true,
                keepConstraints: Boolean = true,
                keepSignatures: Boolean = true,
                keepDocs: Boolean = true,
                keepConfigs: Boolean = true
) {
    if(mixin.legacyWasSet){
        legacy = mixin.legacy
    }
    if(mixin.javaPackageWasSet){
        javaPackage = mixin.javaPackage
    }
    if (keepArgs) {
        args.addOrReplaceAll(mixin.args)
    }
    if (keepInputs) {
        inputs.addOrReplaceAll(mixin.inputs)
    }
    if (keepOutputs) {
        outputs.addOrReplaceAll(mixin.outputs)
    }
    if (keepConstraints) {
        constraints.addAll(mixin.constraints)
    }
    if (keepSignatures) {
        signatures.addAll(mixin.signatures)
    }
    if (keepDocs) {
        doc.addAll(mixin.doc)
    }
    if(keepConfigs){
        configs.addOrReplaceAll(mixin.configs)
    }
}

fun Mixin.useMixin(mixin: Mixin,
                   keepArgs: Boolean = true,
                   keepInputs: Boolean = true,
                   keepOutputs: Boolean = true,
                   keepConstraints: Boolean = true,
                   keepSignatures: Boolean = true,
                   keepDocs: Boolean = true,
                   keepConfigs: Boolean = true) {
    if(mixin.legacyWasSet){
        legacy = mixin.legacy
    }
    if(mixin.javaPackageWasSet){
        javaPackage = mixin.javaPackage
    }
    if (keepArgs) {
        args.addOrReplaceAll(mixin.args)
    }
    if (keepInputs) {
        inputs.addOrReplaceAll(mixin.inputs)
    }
    if (keepOutputs) {
        outputs.addOrReplaceAll(mixin.outputs)
    }
    if (keepConstraints) {
        constraints.addAll(mixin.constraints)
    }
    if (keepSignatures) {
        signatures.addAll(mixin.signatures)
    }
    if (keepDocs) {
        doc.addAll(mixin.doc)
    }
    if(keepConfigs){
        configs.addOrReplaceAll(mixin.configs)
    }
}