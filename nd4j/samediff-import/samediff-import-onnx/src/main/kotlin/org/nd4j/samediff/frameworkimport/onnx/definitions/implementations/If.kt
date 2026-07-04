/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.samediff.frameworkimport.onnx.definitions.implementations

import onnx.Onnx
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.SameDiffNoArgSingleLambda
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.onnx.convertToOnnxDataType
import org.nd4j.samediff.frameworkimport.onnx.convertToOnnxTensor
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A port of if.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/if.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["If"],frameworkName = "onnx")
class If : PreImportHook  {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#non

        val registryCast = mappingRegistry as OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>
        val importGraphCast = importGraph as ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>
        // ONNX If branches may reference outer-scope values IMPLICITLY (no declared
        // subgraph inputs — newer optimum exports like whisper's merged decoder do
        // this; older exports declare them explicitly). The branch is imported into a
        // FRESH SameDiff, so implicit captures resolve to nothing and the import dies
        // ("Variable 'input_ids' required by node ... does not exist"). Materialize
        // the captures as declared branch inputs — dtypes come from the parent sd,
        // where every capture already exists (topological order guarantees the If's
        // dependencies imported first). Declared inputs become placeholders in the
        // branch SameDiff, and invokeGraphOn unifies them with the parent's variables
        // by name — the exact mechanism explicit-input branches already use.
        val wrappedThenBranch = materializeImplicitCaptures(
            attributes["then_branch"] as OnnxIRGraph, sd, registryCast, "${op.name}_then_branch")
        val wrappedElseBranch = materializeImplicitCaptures(
            attributes["else_branch"] as OnnxIRGraph, sd, registryCast, "${op.name}_else_branch")
        // Branch imports bypass OnnxFrameworkImporter.runImport, which normally seeds
        // dynamicVariables with every initializer and Constant-node output so mappers
        // needing COMPILE-TIME values (Unsqueeze axes, Reshape shapes, ...) can read
        // them. Without this, branch-local initializers fail with "must be available
        // as an ONNX initializer constant". Mirror that seeding per branch.
        val thenBranchSubGraph = importGraphCast.importGraph(
            wrappedThenBranch,
            null,
            null, collectBranchConstants(wrappedThenBranch),
            registryCast,
            false
        )

        logBranchOutputStates(thenBranchSubGraph, wrappedThenBranch, "${op.name}/then")
        namespaceBranchInternals(thenBranchSubGraph, sd, "${op.name}/then")
        sd.putSubFunction("${op.name}_then_branch",thenBranchSubGraph)
        val elseBranchSubGraph = importGraphCast.importGraph(
            wrappedElseBranch,
            null,
            null, collectBranchConstants(wrappedElseBranch),
            registryCast,
            false
        )
        namespaceBranchInternals(elseBranchSubGraph, sd, "${op.name}/else")
        sd.putSubFunction("${op.name}_else_branch",elseBranchSubGraph)

        val outputVarName = outputNames[0]

        // Composition is GRAPH BUILDING, not execution: with import-time eager mode
        // on, the interceptor-created switch/flat ops execute as they are created —
        // on frame values that are empty/dead at build time — fail with WARNs, and
        // leave stale EMPTY eager arrays that session execution later mistakes for
        // real values (concat receiving empty shape components). Suspend eager mode
        // for the duration of the ifCond wiring.
        // Eager mode stays suspended for the ENTIRE composition — ifCond AND the
        // multi-output switch/merge wiring below. Any graph-building call that
        // eager-executes here runs on dead/absent frame values and deposits stale
        // arrays (observed: a scoped switch's DEAD side carrying a live [1,1] at
        // session time because post-ifCond wiring ran with eager restored).
        val wasEager = sd.isEagerMode
        if (wasEager) sd.disableEagerMode()
        try {
        val outputVar = sd.ifCond(outputVarName,outputVarName,SameDiffNoArgSingleLambda {
            sd.getVariable(op.inputsToOp[0])
        }, SameDiffNoArgSingleLambda {
            val definedFunction = sd.getFunction("${op.name}_then_branch")
            definedFunction.invokeGraphOn(sd)
        }, SameDiffNoArgSingleLambda {
            val definedFunction = sd.getFunction("${op.name}_else_branch")
            definedFunction.invokeGraphOn(sd)

        })

        // Interception probe: list composed ops that consume the raw capture
        // placeholders directly. ControlFlow's argument interceptor should have
        // rewired every frame-op arg to a switch output — direct consumers other
        // than the switches themselves mean frame gating is not wired.
        run {
            var direct = 0
            for ((opName2, sdo2) in sd.ops) {
                val ins = sdo2.inputsToOp ?: continue
                if ((ins.contains("input_ids") || ins.contains("encoder_hidden_states"))
                        && sdo2.op !is org.nd4j.linalg.api.ops.impl.controlflow.compat.Switch) {
                    direct++
                    if (direct <= 6) {
                        log.warn("If '{}': op '{}' ({}) consumes RAW capture directly: {}",
                                op.name, opName2, sdo2.op?.opName(), ins)
                    }
                }
            }
            log.info("If '{}': {} non-Switch ops consume raw captures directly", op.name, direct)
        }

        val result = linkedMapOf(outputVar.name() to listOf(outputVar))

        // ONNX If declares one node output PER BRANCH OUTPUT, elementwise (then[i] /
        // else[i] -> output[i]). ifCond wires only output 0; historically the rest
        // were silently dropped (masked while branch names leaked unprefixed).
        // Wire each remaining output with the exact switch/merge pattern ControlFlow
        // uses: then side through switch[1] (predicate true), else through switch[0].
        if (outputNames.size > 1) {
            val pred = sd.getVariable(op.inputsToOp[0])
            val thenOuts = wrappedThenBranch.graphDef().outputList.map { it.name.replace(":0", "") }
            val elseOuts = wrappedElseBranch.graphDef().outputList.map { it.name.replace(":0", "") }
            for (i in 1 until outputNames.size) {
                val thenName = "${op.name}/then/${thenOuts[i]}"
                val elseName = "${op.name}/else/${elseOuts[i]}"
                if (!sd.hasVariable(thenName) || !sd.hasVariable(elseName)) {
                    throw IllegalStateException("If '${op.name}' output ${i} ('${outputNames[i]}'): " +
                            "branch outputs '$thenName' / '$elseName' not found after composition")
                }
                val thenSwitched = sd.switchOp(sd.getVariable(thenName), pred)[1]
                val elseSwitched = sd.switchOp(sd.getVariable(elseName), pred)[0]
                val merged = sd.merge(thenSwitched, elseSwitched)
                sd.renameVariable(merged.name(), outputNames[i])
                result[outputNames[i]] = listOf(sd.getVariable(outputNames[i]))
            }
            log.info("If '{}': wired {} additional branch outputs via switch/merge", op.name, outputNames.size - 1)
            // Wiring audit: name each output's merge and the branch variables feeding
            // its two switches, so mispairings (then[i]/else[i] order skew between the
            // exported branches) or missing sides are readable from one import log.
            for (i in 1 until outputNames.size) {
                val outMeta = sd.variables[outputNames[i]]
                val mergeOp = outMeta?.outputOfOp?.let { sd.ops[it] }
                log.info("If '{}' output[{}] '{}': merge='{}' inputs={} (then[{}]='{}', else[{}]='{}')",
                        op.name, i, outputNames[i], outMeta?.outputOfOp,
                        mergeOp?.inputsToOp, i, thenOuts.getOrNull(i), i, elseOuts.getOrNull(i))
            }
        }

        // Post-composition linkage audit: every branch-scoped variable in the PARENT
        // must know its producing op AND that op must be registered in the parent.
        // (Branch-side linkage validated clean; a break here means invokeGraphOn
        // dropped it during cloning.)
        var varMissingProducer = 0
        var opNotRegistered = 0
        for ((name, meta) in sd.variables) {
            if (!name.startsWith("${op.name}/")) continue
            val producer = meta.outputOfOp
            if (producer == null) {
                if (meta.variable.variableType == org.nd4j.autodiff.samediff.VariableType.ARRAY) {
                    varMissingProducer++
                    if (varMissingProducer <= 5) {
                        log.warn("If '{}' composition: ARRAY variable '{}' has NO outputOfOp in parent", op.name, name)
                    }
                }
            } else if (!sd.opExists(producer)) {
                opNotRegistered++
                if (opNotRegistered <= 5) {
                    log.warn("If '{}' composition: variable '{}' producer op '{}' NOT registered in parent",
                            op.name, name, producer)
                }
            }
        }
        if (varMissingProducer > 0 || opNotRegistered > 0) {
            log.warn("If '{}' composition audit: {} ARRAY vars missing producer, {} producer ops unregistered",
                    op.name, varMissingProducer, opNotRegistered)
        }

        return result
        } finally {
            if (wasEager) sd.enableEagerMode()
        }
    }

    /**
     * Rewrite a branch subgraph so every implicit outer-scope capture becomes a
     * DECLARED graph input. Returns the original graph untouched when there are no
     * implicit captures (the explicit-input export style).
     */
    private fun materializeImplicitCaptures(
        branch: OnnxIRGraph,
        parent: SameDiff,
        registry: OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>,
        branchName: String
    ): OnnxIRGraph {
        val graphDef = branch.graphDef()
        val produced = HashSet<String>()
        val referenced = HashSet<String>()
        collectNames(graphDef, produced, referenced)
        graphDef.initializerList.forEach { produced.add(it.name) }
        graphDef.inputList.forEach { produced.add(it.name) }

        val captures = referenced
            .map { it.replace(":0", "") }
            .filter { it.isNotEmpty() && it !in produced }
            .distinct()
            .filter { parent.hasVariable(it) }
            .sorted()
        if (captures.isEmpty()) {
            return branch
        }

        val rebuilt = graphDef.toBuilder()
        var asInitializers = 0
        var asInputs = 0
        for (capture in captures) {
            val variable = parent.getVariable(capture)
            val arr = variable.arr
            // Captures with KNOWN VALUES in the parent (weights, folded constants,
            // axes/shape scalars) must enter the branch as INITIALIZERS: op mappers
            // fold them at import time (Unsqueeze axes, Reshape shapes, ...) and a
            // runtime placeholder there fails with "must be available as an ONNX
            // initializer constant". Only genuinely runtime tensors (placeholders —
            // input_ids, encoder_hidden_states, past KV) become declared inputs,
            // unified with the parent by name at invokeGraphOn.
            if (arr != null && variable.variableType != org.nd4j.autodiff.samediff.VariableType.PLACEHOLDER) {
                rebuilt.addInitializer(convertToOnnxTensor(arr, capture))
                asInitializers++
            } else {
                // Carry the parent's declared shape (dynamic dims as -1): a RANKLESS
                // placeholder poisons downstream shape math — shape_of() on it yields
                // EMPTY, which then flows through gathers/reshapes/concats as "legal"
                // empty values instead of resolvable dims.
                val tensorType = Onnx.TypeProto.Tensor.newBuilder()
                        .setElemType(convertToOnnxDataType(variable.dataType()).number)
                val parentShape = variable.shape ?: variable.placeholderShape()
                if (parentShape != null && parentShape.isNotEmpty()) {
                    val shapeProto = Onnx.TensorShapeProto.newBuilder()
                    for (d in parentShape) {
                        shapeProto.addDim(Onnx.TensorShapeProto.Dimension.newBuilder().setDimValue(d).build())
                    }
                    tensorType.setShape(shapeProto.build())
                }
                rebuilt.addInput(Onnx.ValueInfoProto.newBuilder()
                        .setName(capture)
                        .setType(Onnx.TypeProto.newBuilder()
                                .setTensorType(tensorType.build())
                                .build())
                        .build())
                asInputs++
            }
        }
        log.info("If branch '{}': materialized {} implicit outer-scope captures " +
                "({} constant initializers, {} runtime inputs)",
                branchName, captures.size, asInitializers, asInputs)
        return OnnxIRGraph(rebuilt.build(), registry)
    }

    /** Branch-side truth for every declared graph output: producer, type, stored array. */
    private fun logBranchOutputStates(branch: SameDiff, wrapped: OnnxIRGraph, prefix: String) {
        for (out in wrapped.graphDef().outputList) {
            val name = out.name.replace(":0", "")
            if (!branch.hasVariable(name)) {
                log.warn("BRANCH-OUT '{}' [{}]: VARIABLE DOES NOT EXIST in branch sd", prefix, name)
                continue
            }
            val meta = branch.variables[name]
            val v = branch.getVariable(name)
            val arr = try { v.arr } catch (e: Exception) { null }
            var protoProducer = "NONE"
            for (n in wrapped.graphDef().nodeList) {
                if (n.outputList.any { o -> o.replace(":0", "") == name }) {
                    protoProducer = n.opType + "(" + n.inputList.joinToString(",") + ")"
                    break
                }
            }
            log.info("BRANCH-OUT '{}' [{}]: type={} producer={} arr={} protoNode={}",
                    prefix, name, v.variableType, meta?.outputOfOp,
                    if (arr == null) "null" else java.util.Arrays.toString(arr.shape()) + (if (arr.isEmpty) "(EMPTY)" else ""),
                    protoProducer)
        }
    }

    /**
     * Rename branch-INTERNAL variables that collide with names already present in the
     * enclosing graph. invokeGraphOn's rename-on-collision creates fresh names while
     * op argument rebinding still resolves the ORIGINAL names — crossing wires between
     * the branch and outer scopes (observed as reshape ops receiving another node's
     * inputs). Removing the collisions upfront means invokeGraphOn never renames:
     * capture PLACEHOLDERS keep their names (unification with the outer scope is the
     * point), everything else moves into a per-branch namespace.
     */
    private fun namespaceBranchInternals(branch: SameDiff, parent: SameDiff, prefix: String) {
        // Rename ALL branch internals, not only parent-colliding ones: the then and
        // else branches of a merged export share hundreds of internal names with EACH
        // OTHER, and the second branch composes after the first's copies are already
        // in the parent — reintroducing rename-on-collision cross-wiring between the
        // frames. Capture PLACEHOLDERS keep their names (outer-scope unification is
        // intentional); everything else moves into the branch namespace.
        val toRename = branch.variables()
            .filter { v -> v.variableType != org.nd4j.autodiff.samediff.VariableType.PLACEHOLDER }
            .map { it.name() }
        if (toRename.isEmpty()) return
        log.info("If branch namespace '{}': renaming {} branch-internal variables into the branch scope",
                prefix, toRename.size)
        for (name in toRename) {
            branch.renameVariable(name, "$prefix/$name")
        }

        // Op OUTPUTS must be ARRAY-typed (computed activations). Eager-mode branch
        // import can materialize op outputs as VARIABLE-with-array — e.g.
        // Identity(unfed shaped placeholder) baking an empty [0,...] — which
        // composition then treats as a VALUE-bearing variable: the junk array is
        // associated into the parent and served from initialization on every call,
        // permanently shadowing the real computation (whisper: encoder presents).
        var coerced = 0
        for (v in branch.variables()) {
            val meta = branch.variables[v.name()] ?: continue
            if (meta.outputOfOp != null
                    && v.variableType == org.nd4j.autodiff.samediff.VariableType.VARIABLE) {
                v.setVariableType(org.nd4j.autodiff.samediff.VariableType.ARRAY)
                coerced++
            }
        }
        if (coerced > 0) {
            log.info("If branch '{}': coerced {} eager-materialized op outputs from VARIABLE to ARRAY", prefix, coerced)
        }

        // Namespace OP OWN-NAMES as well: then/else branches of a merged export share
        // nearly all op names, and colliding own-names break composition —
        // addOutgoingFor early-returns ("Outgoing arguments already declared") on the
        // second branch's clones, leaving their outputs with NO producer link
        // (712 dead variables observed on whisper's else branch), and
        // opExists/putOpForId overwrites the first branch's registration.
        val opNames = branch.ops.keys.toList()
        for (old in opNames) {
            val newName = "$prefix/$old"
            val sdo = branch.ops.remove(old) ?: continue
            sdo.name = newName
            sdo.op?.ownName = newName
            branch.ops[newName] = sdo
        }
        for ((_, meta) in branch.variables) {
            meta.outputOfOp?.let { if (!it.startsWith("$prefix/")) meta.outputOfOp = "$prefix/$it" }
            meta.inputsForOp = meta.inputsForOp?.map {
                if (it.startsWith("$prefix/")) it else "$prefix/$it" }?.toMutableList()
            meta.controlDepsForOp = meta.controlDepsForOp?.map {
                if (it.startsWith("$prefix/")) it else "$prefix/$it" }?.toMutableList()
            meta.controlDeps = meta.controlDeps?.map {
                if (it.startsWith("$prefix/")) it else "$prefix/$it" }?.toMutableList()
        }

        // Validate producer linkage survived the rename: a variable whose op link
        // breaks imports fine but can never be produced at execution ("VALUE NOT
        // FOUND IN CONTEXT" nulls in the ACTIVE frame).
        var broken = 0
        for (v in branch.variables()) {
            val meta = branch.variables[v.name()] ?: continue
            val producer = meta.outputOfOp
            if (producer != null) {
                val op = branch.ops[producer]
                if (op == null || op.outputsOfOp == null || !op.outputsOfOp.contains(v.name())) {
                    broken++
                    if (broken <= 5) {
                        log.warn("If branch '{}': rename broke producer linkage for '{}' " +
                                "(outputOfOp='{}', op outputs={})",
                                prefix, v.name(), producer, op?.outputsOfOp)
                    }
                }
            }
        }
        if (broken > 0) {
            log.warn("If branch '{}': {} variables have BROKEN producer linkage after rename", prefix, broken)
        }
    }

    /**
     * Seed a branch's dynamicVariables the way OnnxFrameworkImporter.runImport does
     * for top-level graphs: every initializer plus every Constant-node "value" output,
     * so import-time value consumers (axes, shapes) resolve inside the branch.
     */
    private fun collectBranchConstants(branch: OnnxIRGraph): MutableMap<String, Onnx.TensorProto> {
        val constants = mutableMapOf<String, Onnx.TensorProto>()
        val graphDef = branch.graphDef()
        for (initializer in graphDef.initializerList) {
            constants.putIfAbsent(initializer.name, initializer)
        }
        for (node in graphDef.nodeList) {
            if (node.opType == "Constant") {
                val tensor = node.attributeList.firstOrNull { it.name == "value" }?.t ?: continue
                for (outputName in node.outputList) {
                    constants.putIfAbsent(outputName, tensor)
                }
            }
        }
        return constants
    }

    /** Collect produced (node outputs) and referenced (node inputs) names, recursing into subgraph attributes. */
    private fun collectNames(graphDef: Onnx.GraphProto, produced: MutableSet<String>, referenced: MutableSet<String>) {
        for (node in graphDef.nodeList) {
            node.outputList.forEach { produced.add(it.replace(":0", "")) }
            node.inputList.forEach { referenced.add(it.replace(":0", "")) }
            for (attr in node.attributeList) {
                if (attr.hasG()) {
                    // Names produced or declared INSIDE a nested subgraph are local to it.
                    attr.g.initializerList.forEach { produced.add(it.name) }
                    attr.g.inputList.forEach { produced.add(it.name) }
                    collectNames(attr.g, produced, referenced)
                }
                for (nested in attr.graphsList) {
                    nested.initializerList.forEach { produced.add(it.name) }
                    nested.inputList.forEach { produced.add(it.name) }
                    collectNames(nested, produced, referenced)
                }
            }
        }
    }

    companion object {
        private val log = org.slf4j.LoggerFactory.getLogger(If::class.java)
    }
}