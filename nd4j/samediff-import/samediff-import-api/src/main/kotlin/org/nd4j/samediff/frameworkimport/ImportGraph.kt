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
package org.nd4j.samediff.frameworkimport

import org.apache.commons.collections4.set.ListOrderedSet
import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.autodiff.samediff.internal.Variable
import org.nd4j.common.base.Preconditions
import org.nd4j.common.io.ReflectionUtils
import org.nd4j.imports.converters.DifferentialFunctionClassHolder
import org.nd4j.imports.graphmapper.OpImportFilter
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.BaseOp
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.impl.controlflow.compat.BaseCompatOp
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.debug.VariableOriginTracer
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.registry.OpRegistryHolder
import org.nd4j.samediff.frameworkimport.runner.DefaultImportRunner
import org.nd4j.samediff.frameworkimport.runner.ImportRunner
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.util.*

import mu.KotlinLogging
import org.nd4j.linalg.api.ndarray.INDArray
import kotlin.collections.HashMap

open class ImportGraph <GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTR_DEF_TYPE : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {

    private val logger = KotlinLogging.logger {}
    val defaultRunner = DefaultImportRunner<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>()
    private var isTracingEnabled = true

    data class TopologicalSortResult<NODE_TYPE: GeneratedMessageV3, TENSOR_TYPE: GeneratedMessageV3,
            ATTR_DEF_TYPE: GeneratedMessageV3, ATTR_VALUE_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(
        val sortedNodes: List<IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>>,
        val sortedVariables: List<String>,
        val nodeNameMapping: Map<IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>, String>,
        val variableToProducer: Map<String, String>,
        val variableToConsumers: Map<String, Set<String>>,
        val nodeDependencies: Map<String, Set<String>>,
        val variableDependencies: Map<String, Set<String>>
    )

    fun isControlDep(name: String): Boolean = name.startsWith("^")
    fun stripControl(name: String): String = if (name.startsWith("^")) name.substring(1) else name
    private fun stripVarSuffix(name: String): String = if (name.endsWith(":0")) name.substring(0, name.length - 2) else name

    fun performTopologicalSort(
        irGraph: IRGraph<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>,
        dynamicVariables: MutableMap<String, TENSOR_TYPE>
    ): TopologicalSortResult<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE> {

        logger.info("=== TOPOLOGICAL SORT ===")

        val originalNodeList = irGraph.nodeList()
        val nodeNameMapping = mutableMapOf<IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>, String>()
        val allVariables = mutableSetOf<String>()
        val varProducer = mutableMapOf<String, String>()
        val varConsumers = mutableMapOf<String, MutableSet<String>>()
        val nodeInputs = mutableMapOf<String, MutableSet<String>>()
        val nodeOutputs = mutableMapOf<String, MutableSet<String>>()

        // Build basic mappings
        originalNodeList.forEachIndexed { index, node ->
            val assignedName = if (node.nodeName().isEmpty()) "${node.opName()}_$index" else node.nodeName()
            nodeNameMapping[node] = assignedName

            nodeInputs[assignedName] = mutableSetOf()
            nodeOutputs[assignedName] = mutableSetOf()

            // Track outputs
            for (i in 0 until node.numOutputs()) {
                val output = stripVarSuffix(node.outputAt(i))
                nodeOutputs[assignedName]!!.add(output)
                allVariables.add(output)
                val previousProducer = varProducer.put(output, assignedName)
                if (previousProducer != null && previousProducer != assignedName) {
                    // Two nodes claiming one output name silently corrupts the sort:
                    // consumers bind to the LAST writer and the real producer can be
                    // ordered after them ("Variable X does not exist" at import).
                    logger.warn("DUPLICATE OUTPUT NAME '{}' produced by both '{}' and '{}' — " +
                            "dependency edges will bind to the latter", output, previousProducer, assignedName)
                }
            }

            // Track inputs (non-control dependencies only)
            for (i in 0 until node.numInputs()) {
                val rawInput = node.inputAt(i)
                if (!isControlDep(rawInput)) {
                    val input = stripVarSuffix(stripControl(rawInput))
                    nodeInputs[assignedName]!!.add(input)
                    allVariables.add(input)
                    if (!varConsumers.containsKey(input)) varConsumers[input] = mutableSetOf()
                    varConsumers[input]!!.add(assignedName)
                }
            }
        }

        // Kahn's algorithm for topological sort - O(n+e) instead of O(n²)
        val processedNodes = mutableSetOf<String>()
        val sortedNodes = mutableListOf<IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>>()
        val nodesByName = originalNodeList.associateBy { nodeNameMapping[it]!! }

        // Build node-to-node dependency graph and in-degree counts
        val nodeDeps = mutableMapOf<String, MutableSet<String>>() // node -> set of nodes it depends on
        val nodeRdeps = mutableMapOf<String, MutableSet<String>>() // node -> set of nodes that depend on it
        val inDegree = mutableMapOf<String, Int>()

        for (node in originalNodeList) {
            val nodeName = nodeNameMapping[node]!!
            nodeDeps[nodeName] = mutableSetOf()
            nodeRdeps.getOrPut(nodeName) { mutableSetOf() }
        }

        for (node in originalNodeList) {
            val nodeName = nodeNameMapping[node]!!
            val inputVars = nodeInputs[nodeName]!!
            for (inputVar in inputVars) {
                val producerNode = varProducer[inputVar]
                if (producerNode != null && producerNode != nodeName) {
                    nodeDeps[nodeName]!!.add(producerNode)
                    nodeRdeps.getOrPut(producerNode) { mutableSetOf() }.add(nodeName)
                }
            }
            inDegree[nodeName] = nodeDeps[nodeName]!!.size
        }

        // Initialize queue with nodes that have no dependencies
        val queue = ArrayDeque<String>()
        for (node in originalNodeList) {
            val nodeName = nodeNameMapping[node]!!
            if (inDegree[nodeName] == 0) {
                queue.add(nodeName)
            }
        }

        // Process queue
        while (queue.isNotEmpty()) {
            val nodeName = queue.removeFirst()
            sortedNodes.add(nodesByName[nodeName]!!)
            processedNodes.add(nodeName)

            // Reduce in-degree for dependents
            for (dependent in nodeRdeps[nodeName] ?: emptySet()) {
                val newDegree = inDegree[dependent]!! - 1
                inDegree[dependent] = newDegree
                if (newDegree == 0) {
                    queue.add(dependent)
                }
            }
        }

        // Add any remaining nodes (shouldn't happen in a proper DAG)
        val unprocessedNodes = mutableListOf<String>()
        originalNodeList.forEach { node ->
            val nodeName = nodeNameMapping[node]!!
            if (nodeName !in processedNodes) {
                sortedNodes.add(node)
                unprocessedNodes.add(nodeName)
            }
        }
        // Unprocessed nodes mean a dependency CYCLE (or edges to nonexistent
        // producers): they get appended in original order, which places producers
        // AFTER their consumers and later fails the import with "Variable X does
        // not exist". Name the blocking edges so the cycle is directly readable.
        if (unprocessedNodes.isNotEmpty()) {
            logger.warn("TOPOLOGICAL SORT INCOMPLETE: {} nodes never reached in-degree 0 " +
                    "(dependency cycle) — appended in original order (first 10: {})",
                unprocessedNodes.size, unprocessedNodes.take(10))
            unprocessedNodes.take(5).forEach { stuck ->
                val blocking = nodeDeps[stuck]?.filter { it !in processedNodes } ?: emptyList()
                logger.warn("  stuck node '{}' blocked by unprocessed deps: {}", stuck, blocking.take(8))
            }
        }

        // Create variable order based on processing order
        val sortedVariables = mutableListOf<String>()
        val processedVars = mutableSetOf<String>()

        sortedNodes.forEach { node ->
            val nodeName = nodeNameMapping[node]!!
            nodeOutputs[nodeName]?.forEach { outputVar ->
                if (outputVar !in processedVars) {
                    sortedVariables.add(outputVar)
                    processedVars.add(outputVar)
                }
            }
        }

        // Add any remaining variables
        allVariables.filter { it !in processedVars }.forEach { variable ->
            sortedVariables.add(variable)
        }

        // Build final dependency maps
        val nodeDependencies = mutableMapOf<String, Set<String>>()
        val varInputs = mutableMapOf<String, MutableSet<String>>()

        originalNodeList.forEach { node ->
            val nodeName = nodeNameMapping[node]!!
            val deps = mutableSetOf<String>()
            nodeInputs[nodeName]?.forEach { inputVar ->
                val producerNode = varProducer[inputVar]
                if (producerNode != null && producerNode != nodeName) {
                    deps.add(producerNode)
                }
            }
            nodeDependencies[nodeName] = deps
        }

        allVariables.forEach { varName ->
            varInputs[varName] = mutableSetOf()
            val producerNode = varProducer[varName]
            if (producerNode != null) {
                nodeInputs[producerNode]?.forEach { inputVar ->
                    if (inputVar != varName) {
                        varInputs[varName]!!.add(inputVar)
                    }
                }
            }
        }

        // Verify critical ordering
        val equalPos = sortedNodes.indexOfFirst { nodeNameMapping[it] == "/Equal" }
        val wherePos = sortedNodes.indexOfFirst { nodeNameMapping[it] == "/Where" }
        if (equalPos >= 0 && wherePos >= 0) {
            logger.info("Critical verification: /Equal at position $equalPos, /Where at position $wherePos")
            if (equalPos >= wherePos) {
                logger.error("ORDERING ERROR: /Equal should come before /Where!")
            }
        }

        logger.info("SORT COMPLETE: ${sortedVariables.size} variables, ${sortedNodes.size} nodes")

        return TopologicalSortResult(
            sortedNodes = sortedNodes,
            sortedVariables = sortedVariables,
            nodeNameMapping = nodeNameMapping,
            variableToProducer = varProducer,
            variableToConsumers = varConsumers.mapValues { it.value.toSet() },
            nodeDependencies = nodeDependencies,
            variableDependencies = varInputs.mapValues { it.value.toSet() }
        )
    }

    fun <GRAPH_TYPE: GeneratedMessageV3,
            NODE_TYPE: GeneratedMessageV3,
            OP_DEF_TYPE: GeneratedMessageV3,
            TENSOR_TYPE: GeneratedMessageV3,
            ATTRIBUTE_TYPE: GeneratedMessageV3,
            ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
            DATA_TYPE : ProtocolMessageEnum> importInfoForEachNodeInGraph (
        graph: IRGraph<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>,
        dynamicVariables: MutableMap<String, TENSOR_TYPE>)
            :  Map<String,Pair<MappingContext<GRAPH_TYPE,
            NODE_TYPE, OP_DEF_TYPE,
            TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>,OpNamespace.OpDescriptor>> {

        val opMappingRegistry = OpRegistryHolder.opMappingRegistryForName<GRAPH_TYPE,
                NODE_TYPE,
                OP_DEF_TYPE,
                TENSOR_TYPE,
                ATTRIBUTE_TYPE,
                ATTRIBUTE_VALUE_TYPE,
                DATA_TYPE>(graph.frameworkName())

        val ret = HashMap<String,Pair<MappingContext<GRAPH_TYPE,
                NODE_TYPE, OP_DEF_TYPE,
                TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE,
                DATA_TYPE>,OpNamespace.OpDescriptor>>()

        graph.nodeList().forEach { node ->
            val name = node.nodeName()
            val opMappingProcess =  OpRegistryHolder.lookupOpMappingProcess<
                    GRAPH_TYPE,
                    NODE_TYPE,
                    OP_DEF_TYPE,
                    TENSOR_TYPE,
                    DATA_TYPE,
                    ATTRIBUTE_TYPE,
                    ATTRIBUTE_VALUE_TYPE>(inputFrameworkOpName = node.opName(), inputFrameworkName = graph.frameworkName())
            val opDefLookup = opMappingRegistry.lookupInputFrameworkOpDef(node.opName())
            val mappingContext = graph.createMappingContext(
                opDef = opDefLookup,
                node = graph.nodeByName(node.nodeName()),
                dynamicVariables = dynamicVariables
            )

            val applied = opMappingProcess.applyProcess(mappingContext)
            ret[name] = applied
        }

        return ret
    }

    inner class FuncContextResult<GRAPH_TYPE: GeneratedMessageV3, NODE_TYPE: GeneratedMessageV3, OP_DEF_TYPE: GeneratedMessageV3,
            TENSOR_TYPE: GeneratedMessageV3, ATTR_DEF_TYPE: GeneratedMessageV3, ATTR_VALUE_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>
        (dfInstance: DifferentialFunction,mappingContext: MappingContext<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTR_DEF_TYPE,ATTR_VALUE_TYPE,DATA_TYPE>,
         tensorInputMappings: MutableMap<String,String>) {
        val dfInstance = dfInstance
        val mappingContext = mappingContext
        val tensorInputMappings = tensorInputMappings
    }

    fun createFuncAndContext(opName: String,
                             irGraph: IRGraph<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>,
                             opMappingRegistry: OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE>,
                             sameDiff: SameDiff,
                             nodeName: String,
                             dynamicVariables: MutableMap<String, TENSOR_TYPE>): FuncContextResult<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTR_DEF_TYPE,ATTR_VALUE_TYPE,DATA_TYPE> {

        val opMappingProcess =  opMappingRegistry.lookupOpMappingProcess(opName)
        val nd4jOpName = opMappingProcess.opName()

        val dfInstance = if( DifferentialFunctionClassHolder.getInstance()
                .hasName(nd4jOpName)) DifferentialFunctionClassHolder
            .getInstance(nd4jOpName)
        else DynamicCustomOp.builder(nd4jOpName).build()
        Preconditions.checkState(dfInstance != null, "Could not find class for input framework Ops: %s", opName)
        var df: DifferentialFunction = dfInstance.javaClass.newInstance()

        df.sameDiff = sameDiff
        df.ownName = nodeName

        val opDefLookup = opMappingRegistry.lookupInputFrameworkOpDef(opName)
        val mappingContext = irGraph.createMappingContext(
            opDef = opDefLookup,
            node = irGraph.nodeByName(nodeName),
            dynamicVariables = dynamicVariables
        )

        val tensorInputMappings = HashMap<String, String>()
        opMappingProcess.tensorMappingRules().forEach { tensorMappingRule ->
            tensorInputMappings.putAll(tensorMappingRule.inputArgumentMappings())
        }

        return FuncContextResult(df, mappingContext, tensorInputMappings)
    }

    fun importGraph(
        irGraph: IRGraph<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>,
        importOverride: Map<String?, ImportRunner<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>?>?,
        opFilter: OpImportFilter<GRAPH_TYPE, NODE_TYPE, ATTR_VALUE_TYPE>?,
        dynamicVariables: MutableMap<String, TENSOR_TYPE> = HashMap(),
        opMappingRegistry: OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE>,
        trackVariableChanges: Boolean
    ): SameDiff {

        logger.info("=== GRAPH IMPORT ===")

        val sortResult = performTopologicalSort(irGraph, dynamicVariables)
        logger.info("Processing ${sortResult.sortedNodes.size} nodes in sorted order")

        SameDiff.setGraphBuildingMode(true)

        isTracingEnabled = Nd4j.getEnvironment().isVariableTracingEnabled
        if (isTracingEnabled) {
            VariableOriginTracer.clear()
        }

        // Lazy import info: compute per-node on demand instead of upfront for ALL nodes.
        // This avoids calling applyProcess for every node before we even start the import loop.
        val opMappingRegistryForImport = OpRegistryHolder.opMappingRegistryForName<GRAPH_TYPE,
                NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>(irGraph.frameworkName())
        val importInfoCache = HashMap<String, Pair<MappingContext<GRAPH_TYPE,
                NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE,
                DATA_TYPE>, OpNamespace.OpDescriptor>>()

        fun getImportInfoForNode(nodeName: String, node: IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>): Pair<MappingContext<GRAPH_TYPE,
                NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE,
                DATA_TYPE>, OpNamespace.OpDescriptor> {
            return importInfoCache.getOrPut(nodeName) {
                val opMappingProcess = OpRegistryHolder.lookupOpMappingProcess<
                        GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE>(
                    inputFrameworkOpName = node.opName(), inputFrameworkName = irGraph.frameworkName())
                val opDefLookup = opMappingRegistryForImport.lookupInputFrameworkOpDef(node.opName())
                val mappingContext = irGraph.createMappingContext(
                    opDef = opDefLookup,
                    node = irGraph.nodeByName(nodeName),
                    dynamicVariables = dynamicVariables
                )
                opMappingProcess.applyProcess(mappingContext)
            }
        }

        // Check for controlflow ops by scanning node op names directly (avoids computing importInfo for all nodes)
        var containsControlflow = false
        val controlflowOps = setOf("select","while","enter","if","switch","next_iteration","merge","exit","loop_cond")
        for (node in sortResult.sortedNodes) {
            val opName = node.opName()
            // Check the nd4j op name via the mapping registry
            try {
                val opMappingProcess = OpRegistryHolder.lookupOpMappingProcess<
                        GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE>(
                    inputFrameworkOpName = opName, inputFrameworkName = irGraph.frameworkName())
                if (controlflowOps.contains(opMappingProcess.opName()) || node.isControlflowOp()) {
                    containsControlflow = true
                    break
                }
            } catch (e: Exception) {
                // If we can't look up the op, check the node directly
                if (node.isControlflowOp()) {
                    containsControlflow = true
                    break
                }
            }
        }

        val sd = SameDiff.create().enableEagerMode()

        // First pass: Create all external variables
        dynamicVariables.forEach { (name, ndarray) ->
            val converted = irGraph.convertToNDArray(ndarray)
            sd.`var`(name,converted)
            sd.setEagerArrForVarName(name,converted)
            if (isTracingEnabled) {
                VariableOriginTracer.traceVariableResolution(name, "dynamic_variable", sd.getVariable(name), converted)
            }
        }

        // Second pass: Create constants, placeholders, variables
        sortResult.sortedNodes.forEach { node ->
            val nodeName = sortResult.nodeNameMapping[node]!!
            val opName = node.opName()

            if (irGraph.nodeIsPlaceHolder(node.nodeName())) {
                // Skip if a variable with this name was already registered as a dynamic variable
                // in the first pass. This happens when ONNX initializers overlap with graph inputs
                // (e.g. input.1 appears both as an initializer in dynamicVariables and as a
                // placeholder in the graph) — addInitializersToDynamicVariables may have added it.
                if (sd.hasVariable(nodeName)) {
                    if (isTracingEnabled) {
                        VariableOriginTracer.traceVariableResolution(nodeName, "placeholder_skipped_exists", sd.getVariable(nodeName), null)
                    }
                } else {
                val shape = irGraph.shapeOfInput(node.nodeName())
                val dt = irGraph.dataTypeForVariable(node.nodeName()).nd4jDataType()
                if (shape != null) {
                    sd.placeHolder(nodeName, dt, *shape)
                } else {
                    sd.placeHolder(nodeName, dt)
                }
                if (isTracingEnabled) {
                    VariableOriginTracer.traceVariableResolution(nodeName, "placeholder", sd.getVariable(nodeName), null)
                }
                }
            } else if (irGraph.isConstantOpName(opName)) {
                val arr = irGraph.getConstantArrayForName(nodeName)
                val constantOutputName = if (node.numOutputs() < 1 || irGraph.frameworkName().contains("tensorflow")) nodeName else node.outputAt(0)
                // Skip if a variable with this name was already registered as a dynamic variable
                // in the first pass (e.g. Constant node output added by addInitializersToDynamicVariables).
                if (!sd.hasVariable(constantOutputName)) {
                    if (node.numOutputs() < 1 || irGraph.frameworkName().contains("tensorflow")) {
                        sd.constant(nodeName, arr)
                    } else {
                        sd.constant(node.outputAt(0), arr)
                    }
                }
                if (isTracingEnabled) {
                    VariableOriginTracer.traceVariableResolution(nodeName, "constant", sd.getVariable(nodeName), arr)
                }
            } else if (irGraph.isVariable(nodeName)) {
                val shape = irGraph.shapeOfInput(nodeName)
                val dt = irGraph.dataTypeForVariable(nodeName).nd4jDataType()
                // Skip if variable already exists from first pass dynamic variables
                if (!sd.hasVariable(nodeName)) {
                    if (shape != null) {
                        sd.`var`(nodeName, dt, *shape)
                    } else {
                        sd.`var`(nodeName, dt, -1)
                    }
                }
                if (isTracingEnabled) {
                    VariableOriginTracer.traceVariableResolution(nodeName, "variable", sd.getVariable(nodeName), null)
                }
            }
        }

        // Third pass: Create function contexts for compute nodes
        val nodeNameToFuncContext = HashMap<String,FuncContextResult<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTR_DEF_TYPE,ATTR_VALUE_TYPE,DATA_TYPE>>()

        sortResult.sortedNodes.forEach { node ->
            val assignedName = sortResult.nodeNameMapping[node]!!

            if(!irGraph.isConstantOpName(node.opName()) &&
                !irGraph.nodeIsPlaceHolder(node.nodeName()) &&
                !irGraph.isVariable(assignedName)) {
                val funcAndContext = createFuncAndContext(node.opName(), irGraph, opMappingRegistry, sd, assignedName, dynamicVariables)
                nodeNameToFuncContext[assignedName] = funcAndContext
            }
        }

        val mergeOpsPostProcess: MutableMap<String, String> = HashMap()
        val constControlDeps: MutableMap<String, List<String>> = HashMap()

        // Fourth pass: Process compute nodes in dependency order
        sortResult.sortedNodes.forEachIndexed { index, node ->
            val nodeName = sortResult.nodeNameMapping[node]!!
            val opName = node.opName()

            // Skip nodes that were already processed in earlier passes
            if (irGraph.nodeIsPlaceHolder(node.nodeName()) ||
                irGraph.isConstantOpName(opName) ||
                irGraph.isVariable(nodeName)) {
                return@forEachIndexed
            }

            val importInfoForNode = getImportInfoForNode(nodeName, node)
            val opMappingProcess = OpRegistryHolder.lookupOpMappingProcess<
                    GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE>(
                inputFrameworkOpName = opName, inputFrameworkName = irGraph.frameworkName())

            val funcContextResult = nodeNameToFuncContext[nodeName]!!
            val df = funcContextResult.dfInstance
            val mappingContext = funcContextResult.mappingContext
            val nd4jOpName = df.opName()

            val rawAttrMap = HashMap<String, ATTR_VALUE_TYPE>()
            node.attributeMap().forEach { (name, def) ->
                rawAttrMap[name] = def.internalAttributeValue()
            }

            // Check if node should be skipped
            if (opFilter != null && opFilter.skipOp(node.internalValue(), sd, rawAttrMap, irGraph.internalValue())) {
                return@forEachIndexed
            }

            if (importOverride?.containsKey(nodeName) == true) {
                return@forEachIndexed
            }

            // Process input variables
            var controlDeps: MutableList<String?>? = null
            val numInputs = node.numInputs()
            val inNames: MutableList<String> = ArrayList(numInputs)

            for (i in 0 until numInputs) {
                val origInName = node.inputAt(i)
                var inName = stripControl(origInName)
                if (inName.endsWith(":0")) {
                    inName = inName.substring(0, inName.length - 2)
                }

                // Skip empty input names - ONNX uses "" for optional inputs that aren't provided
                if (inName.isEmpty()) {
                    logger.debug("Skipping empty input at position $i for node '$nodeName' (optional input not provided)")
                    inNames.add("")  // Add empty placeholder to preserve input indices
                    continue
                }

                val isControlDep = isControlDep(origInName)
                if (isControlDep) {
                    if (controlDeps == null) controlDeps = ArrayList()
                    controlDeps.add(inName)
                } else {
                    inNames.add(inName)
                }

                if (!sd.hasVariable(inName) && !isControlDep && inName.isNotEmpty()) {
                    // Try to resolve missing variables
                    var variableResolved = false

                    try {
                        // Check if it's a constant in the IR graph
                        // BUT skip if a node in the graph produces this variable — the node's
                        // computed output should take precedence over an initializer default value.
                        // ONNX models sometimes have initializers that share names with node outputs
                        // (e.g., shape tensors computed by Concat nodes).
                        val hasProducer = sortResult.variableToProducer.containsKey(inName)
                        if (irGraph.hasConstantInitializer(inName) && !hasProducer) {
                            val constantArray = irGraph.getConstantArrayForName(inName)
                            sd.constant(inName, constantArray)
                            if (sd.isEagerMode) {
                                sd.setEagerArrForVarName(inName, constantArray)
                            }
                            if (isTracingEnabled) {
                                VariableOriginTracer.traceVariableResolution(inName, "resolved_constant", sd.getVariable(inName), constantArray)
                            }
                            variableResolved = true
                            logger.debug("Resolved missing variable '$inName' as constant from IR graph")

                        } else if (irGraph.hasNode(inName)) {
                            // Check if it's a constant node in the graph
                            val irNode = irGraph.irNodeByName(inName)
                            if (irGraph.isConstantOpName(irNode.opName())) {
                                val constantArray = irGraph.getConstantArrayForName(inName)
                                if (irNode.numOutputs() < 1 || irGraph.frameworkName().contains("tensorflow")) {
                                    sd.constant(inName, constantArray)
                                } else {
                                    sd.constant(irNode.outputAt(0), constantArray)
                                }
                                if (sd.isEagerMode) {
                                    sd.setEagerArrForVarName(inName, constantArray)
                                }
                                if (isTracingEnabled) {
                                    VariableOriginTracer.traceVariableResolution(inName, "resolved_constant_node", sd.getVariable(inName), constantArray)
                                }
                                variableResolved = true
                                logger.debug("Resolved missing variable '$inName' as constant node from IR graph")
                            }
                        }

                        // Check if it's available in dynamic variables
                        if (!variableResolved && dynamicVariables.containsKey(inName)) {
                            val tensorValue = dynamicVariables[inName]!!
                            val convertedArray = irGraph.convertToNDArray(tensorValue)
                            sd.`var`(inName, convertedArray)
                            if (sd.isEagerMode) {
                                sd.setEagerArrForVarName(inName, convertedArray)
                            }
                            if (isTracingEnabled) {
                                VariableOriginTracer.traceVariableResolution(inName, "resolved_dynamic", sd.getVariable(inName), convertedArray)
                            }
                            variableResolved = true
                            logger.debug("Resolved missing variable '$inName' from dynamic variables")
                        }

                    } catch (e: Exception) {
                        logger.warn("Failed to resolve variable '$inName' from IR graph: ${e.message}")
                        variableResolved = false
                    }

                    // If we still couldn't resolve the variable, fail with diagnostic information
                    if (!variableResolved) {
                        val errorDetails = buildString {
                            appendLine("FATAL: Variable '$inName' required by node '$nodeName' does not exist and could not be resolved!")
                            appendLine("Diagnostic Information:")
                            appendLine("  - Producer node: ${sortResult.variableToProducer[inName] ?: "UNKNOWN"}")
                            val producerName = sortResult.variableToProducer[inName]
                            val producerPos = if (producerName != null)
                                sortResult.sortedNodes.indexOfFirst { sortResult.nodeNameMapping[it] == producerName } else -1
                            appendLine("  - Producer sort position: $producerPos")
                            appendLine("  - Variable sort position: ${sortResult.sortedVariables.indexOf(inName)}")
                            appendLine("  - Current node sort position: ${sortResult.sortedNodes.indexOf(node)}")
                            appendLine("  - Variable consumers: ${sortResult.variableToConsumers[inName] ?: emptySet()}")
                            appendLine("  - Node dependencies: ${sortResult.nodeDependencies[nodeName] ?: emptySet()}")

                            appendLine("Resolution Attempts:")
                            append("  - SameDiff variables: ")
                            if (sd.variableNames().contains(inName)) {
                                appendLine("FOUND (but hasVariable returned false - possible state issue)")
                            } else {
                                appendLine("NOT FOUND")
                            }

                            append("  - IR graph constant initializer: ")
                            try {
                                if (irGraph.hasConstantInitializer(inName)) {
                                    appendLine("FOUND (but resolution failed)")
                                } else {
                                    appendLine("NOT FOUND")
                                }
                            } catch (e: Exception) {
                                appendLine("ERROR checking - ${e.message}")
                            }

                            append("  - IR graph node: ")
                            try {
                                if (irGraph.hasNode(inName)) {
                                    val nodeType = irGraph.irNodeByName(inName).opName()
                                    appendLine("FOUND (op: $nodeType)")
                                } else {
                                    appendLine("NOT FOUND")
                                }
                            } catch (e: Exception) {
                                appendLine("ERROR checking - ${e.message}")
                            }

                            append("  - Dynamic variables: ")
                            if (dynamicVariables.containsKey(inName)) {
                                appendLine("FOUND (but resolution failed)")
                            } else {
                                appendLine("NOT FOUND")
                            }

                            appendLine("Available variables in SameDiff: ${sd.variableNames().take(10)}")
                            if (sd.variableNames().size > 10) {
                                appendLine("  ... and ${sd.variableNames().size - 10} more")
                            }
                        }

                        throw IllegalStateException(errorDetails)
                    }
                }

                val v = sd.variables[inName]
                if (v == null && df is Merge) {
                    mergeOpsPostProcess[df.ownName] = inName
                    continue
                }

                if (v != null && !isControlDep && (v.inputsForOp == null || !v.inputsForOp.contains(nodeName))) {
                    if (v.inputsForOp == null) v.inputsForOp = ArrayList()
                    v.inputsForOp.add(nodeName)
                } else if (v != null && isControlDep) {
                    if (v.controlDepsForOp == null) v.controlDepsForOp = ArrayList()
                    if (!v.controlDepsForOp.contains(nodeName)) {
                        v.controlDepsForOp.add(nodeName)
                    }
                }
            }

            // Set up the operation
            if(df is DynamicCustomOp) {
                val opField = DynamicCustomOp::class.java.getDeclaredField("opName")
                opField.isAccessible = true
                ReflectionUtils.setField(opField,df,nd4jOpName)
            }

            val op = SameDiffOp.builder()
                .name(nodeName)
                .op(df)
                .controlDeps(controlDeps)
                .build()

            // Handle input array resolution
            val resolvedArgInputs = importInfoForNode.second.argDescriptorList.filter {input -> input.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR}
                .sortedBy { argDescriptor -> argDescriptor.argIndex }

            val numInputsToTake = resolvedArgInputs.size

            if(numInputsToTake != inNames.size) {
                when(opMappingProcess.arrayResolutionType()) {
                    MapperNamespace.VariableResolutionType.DIRECT -> {
                        op.inputsToOp = inNames
                    }
                    MapperNamespace.VariableResolutionType.OVERRIDE -> {
                        if(numInputsToTake < inNames.size)
                            op.inputsToOp = inNames.subList(0, numInputsToTake)
                        else if(numInputsToTake > inNames.size) {
                            val inputsAfterOriginal = resolvedArgInputs.size - numInputs
                            val newInputs = mutableListOf<String>()
                            newInputs.addAll(inNames)
                            op.inputsToOp = newInputs
                            resolvedArgInputs.subList(inputsAfterOriginal,resolvedArgInputs.size).forEach { arg ->
                                val newName = sd.generateNewVarName("${op.name}_${arg.name}",0)
                                op.inputsToOp.add(newName)
                                if(!sd.hasVariable(op.inputsToOp[arg.argIndex])) {
                                    if(arg.inputValue != null) {
                                        val tensorValue = arg.inputValue
                                        val nd4jTensor = ndarrayFromNameSpaceTensor(tensorValue)
                                        sd.`var`(op.inputsToOp[arg.argIndex], nd4jTensor)
                                    } else {
                                        throw IllegalArgumentException("No argument value found for op ${op.name} for value arg with name ${arg.name}")
                                    }
                                }
                            }
                        }  else
                            op.inputsToOp = inNames

                        if(numInputsToTake < numInputs && op.op.opName() != "noop") {
                            for(i in numInputsToTake until numInputs) {
                                if(sd.hasVariable(node.inputAt(i))) {
                                    val currInputVar = sd.variables[node.inputAt(i)]!!
                                    currInputVar.inputsForOp.remove(op.name)
                                }
                            }
                        }
                    }
                    MapperNamespace.VariableResolutionType.ERROR_ON_NOT_EQUAL -> {
                        throw IllegalStateException("Number of variable names for node ${mappingContext!!.nodeName()} not exact equal to number of inputs resolved from nd4j op descriptor which was ${resolvedArgInputs.size}")
                    }
                    MapperNamespace.VariableResolutionType.UNRECOGNIZED -> {
                        throw IllegalArgumentException("Illegal type ${opMappingProcess.arrayResolutionType()}")
                    }
                }
            } else
                op.inputsToOp = inNames

            // Process pre-hooks and initialize operation
            val attributes = mappingContext!!.nodeAttributesAsMap()
            var proceedWithInit = true
            mappingContext.relevantPrehookRules().forEach { rule ->
                proceedWithInit = proceedWithInit && rule.preProcess(
                    op, sd, attributes, importInfoForNode.second, node.outputs(), false,
                    opMappingRegistry as OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
                    this as ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
                    dynamicVariables as Map<String,GeneratedMessageV3>
                ).proceedWithInit
            }

            if(proceedWithInit && !sd.ops.containsKey(nodeName))
                sd.ops[nodeName] = op

            if(proceedWithInit)
                defaultRunner.initAttributes(df, sd, importInfoForNode)

            // Process post-hooks
            mappingContext.relevantPosthookRules().forEach { rule ->
                rule.postProcess(op, sd, attributes, importInfoForNode.second,node.outputs())
            }

            // Create output variables
            if(proceedWithInit) {
                val newInNames = sd.ops[nodeName]!!.inputsToOp
                val newInDtypes: MutableList<DataType> = ArrayList(newInNames.size)

                if (df is Merge) {
                    val v1 = sd.getVariable(newInNames[0])
                    val v2 = sd.getVariable(newInNames[1])
                    val dt1 = if (v1 == null) v2!!.dataType() else v1.dataType()
                    val dt2 = if (v2 == null) v1!!.dataType() else v2.dataType()
                    newInDtypes.add(dt1)
                    newInDtypes.add(dt2)
                }
                else {
                    for (s in newInNames) {
                        val v = sd.getVariable(s)
                        newInDtypes.add(v.dataType())
                    }
                }

                val outputDataTypes = df.calculateOutputDataTypes(newInDtypes)

                val numOutputs = outputDataTypes.size

                if(numOutputs < 1 &&  nd4jOpName != "noop") {
                    throw IllegalStateException("Op $nd4jOpName does not have any outputs!")
                }

                val outSDVars = arrayOfNulls<SDVariable>(numOutputs)
                val outVars = arrayOfNulls<Variable>(numOutputs)
                val outNames: MutableList<String> = ArrayList(numOutputs)

                for (i in 0 until numOutputs) {
                    val dt = outputDataTypes[i]
                    val varName = node.outputAt(i)

                    outSDVars[i] = if(sd.hasVariable(varName)) {
                        sd.getVariable(varName)
                    } else {
                        sd.`var`(varName, VariableType.ARRAY, null, dt)
                    }
                    outNames.add(varName)

                    outSDVars[i]!!.creator = df

                    if(sd.variables.containsKey(varName)) {
                        outVars[i] = sd.variables[varName]
                        if(outVars[i]!!.variable == null)
                            outVars[i]!!.variable = outSDVars[i]

                        if(outVars[i]!!.outputOfOp == null) {
                            outVars[i]!!.outputOfOp = nodeName
                        }
                    }
                    else {
                        outVars[i] = Variable.builder()
                            .name(varName)
                            .variable(outSDVars[i])
                            .inputsForOp(null)
                            .controlDepsForOp(null)
                            .controlDepsForVar(null)
                            .outputOfOp(nodeName)
                            .build()
                        sd.variables[varName] = outVars[i]
                    }

                    if (isTracingEnabled) {
                        VariableOriginTracer.traceVariableResolution(varName, nodeName, outSDVars[i], null)
                    }
                }

                sd.ops[nodeName]!!.outputsOfOp = outNames

                if(sd.isEagerMode && !containsControlflow && df !is BaseCompatOp) {
                    when(val operation = op.op)  {
                        is DynamicCustomOp -> {
                            operation.outputVariables = outSDVars
                            operation.computeArrays()
                        }
                        is BaseOp -> {
                            operation.computeVariables(outSDVars)
                        }
                    }
                }
            }
        }

        // Post-processing: Handle control dependencies and cleanup
        for ((varName, cdOpNames) in constControlDeps) {
            sd.variables[varName]!!.controlDeps = cdOpNames
            for (s in cdOpNames) {
                val sdo = sd.ops[s]
                if(sd.ops.containsKey(s)) {
                    if (sdo!!.controlDepFor == null) sdo.controlDepFor = ArrayList()
                    val l = sdo.controlDepFor
                    if (!l.contains(s)) l.add(varName)
                }
            }
        }

        for ((key, value) in mergeOpsPostProcess) {
            val v = sd.variables[value]
            if(v != null) {
                if ( v!!.inputsForOp == null) v.inputsForOp = ArrayList()
                v.inputsForOp.add(key)
            }
        }

        sd.variables.forEach { (varName, variable) ->
            if (variable.outputOfOp == null && variable.variable.variableType == VariableType.ARRAY) {
                sd.ops.forEach { (opName, op) ->
                    if (op.outputsOfOp.contains(varName) && variable.outputOfOp == null) {
                        variable.outputOfOp = opName
                    }
                }
            }
        }

        logger.info("=== IMPORT COMPLETE ===")
        SameDiff.setGraphBuildingMode(false)

        return sd
    }
}