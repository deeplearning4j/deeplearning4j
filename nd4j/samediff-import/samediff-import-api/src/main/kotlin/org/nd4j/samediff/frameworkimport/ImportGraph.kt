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
import org.nd4j.samediff.frameworkimport.context.MappingContext
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

/**
 * Core import class for running model import for any framework.
 * This should be paired with an [OpMappingRegistry]
 * and a set of classes implemented in protobuf that extend [GeneratedMessageV3]
 * and [ProtocolMessageEnum] respectively.
 *
 * The end result with these abstractions is direct interop with a file format's schema
 * convertable to primitives like Nd4j's [INDArray] and [SameDiff]
 *
 * @author Adam Gibson
 *
 */
open class ImportGraph <GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTR_DEF_TYPE : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {

    private val logger = KotlinLogging.logger {}

    val defaultRunner =
        DefaultImportRunner<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>()



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

    /**
     * @return True if the specified name represents a control dependency (starts with "^")
     */
    fun isControlDep(name: String): Boolean {
        return name.startsWith("^")
    }

    /**
     * @return The specified name without the leading "^" character (if any) that appears for control dependencies
     */
    fun stripControl(name: String): String {
        return if (name.startsWith("^")) {
            name.substring(1)
        } else name
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
            .getInstance()
            .getInstance(nd4jOpName)
        else DynamicCustomOp.builder(nd4jOpName).build()
        Preconditions.checkState(dfInstance != null, "Could not find class for input framework Ops: %s", opName)
        var df: DifferentialFunction = try {
            dfInstance.javaClass.newInstance()
        } catch (t: Throwable) {
            //Should never happen because function was already created via no-arg constructor earlier
            throw RuntimeException(t)
        }

        df.sameDiff = sameDiff
        df.ownName = nodeName

        /**
         * Note that ndarrays actually need to be reordered here when input indices aren't equal to what's in the original framework.
         * We should potentially run the import process sooner and compute the input name
         * ordering from that instead.
         */
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


    /**
     * Import a Graph based on a {@link IRGraph} model from a GraphDef, with optional import overrides
     *
     * @param irGraph       IRGraph reflecting the needed model import
     * @param importOverride Optional import override for specific ops, keyed by op name
     * @param opFilter       Optional filter - ops to exclude/ignore
     * @return Imported model
     */
    fun importGraph(irGraph: IRGraph<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>,
                    importOverride: Map<String?, ImportRunner<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>?>?,
                    opFilter: OpImportFilter<GRAPH_TYPE, NODE_TYPE, ATTR_VALUE_TYPE>?,
                    dynamicVariables: MutableMap<String, TENSOR_TYPE> = HashMap(),
                    opMappingRegistry:
                    OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE,
                            DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE>): SameDiff {




        /*
        First, build an in-memory representation of the graph that allows us to build the graph incrementally
        If we can build the graph incrementally, we can make sure that the added variables are set up with the correct
        datatype and (once implemented) greedy shape inference
         */
        val variablesAdded: MutableList<String> = ArrayList()
        val opsAdded: MutableList<String> = ArrayList()
        val opsImported: MutableList<String> = ArrayList()
        val opsRemoved: MutableList<String> = ArrayList()

        val availableToAddSet = LinkedHashSet<String>() //TODO maybe unnecessary?
        val availableToAdd: Queue<IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>> = LinkedList()
        val remainingNodes: MutableMap<String, IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>> =
            HashMap() //All other nodes, not in availableToAdd
        val nodeInputTo: MutableMap<String, ListOrderedSet<String>> =
            HashMap() // For op x -> y, x is key, y is value. Note that these are OP names not VARIABLE names
        var nNodes: Int
        val importInfo = irGraph.importInfoForEachNode(dynamicVariables = dynamicVariables)
        var containsControlflow = false
        val controlflowOps = setOf("select","while","enter","if","switch","next_iteration","merge","exit","loop_cond")
        for (it in importInfo.values) {
            if (controlflowOps.contains(it.second.name) || it.first.irNode().isControlflowOp()) {
                containsControlflow = true
                break

            }
        }
        //First, add any constants, placeholders, and zero-input ops
        //note: we enable eager mode here for dynamic variable resolution
        val sd = SameDiff.create().enableEagerMode()
        val convertedDynamic = HashMap<String,INDArray>()

        if(dynamicVariables != null) {
            //declare as variables
            dynamicVariables.forEach { (name, ndarray) ->
                val converted = irGraph.convertToNDArray(ndarray)
                /**
                 * TODO: convert placeholders to proper data types
                 * with checking. It appears not all dyanmicVariables will match expected data type.
                 */
                if(!sd.hasVariable(name))
                    sd.`var`(name,converted)
                sd.setEagerArrForVarName(name,converted)
                convertedDynamic[name] = converted
            }
        }



        /**
         * Now the nodes in the graph may change after running an import process.
         * Run an import process first before proceeding to process all the nodes in the graph
         */
        val originalNodeList = irGraph.nodeList()
        val nodeNameToFuncContext = HashMap<String,FuncContextResult<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTR_DEF_TYPE,ATTR_VALUE_TYPE,DATA_TYPE>>()
        originalNodeList.forEach { node ->
            if(!irGraph.isConstant(node.opName()) && !irGraph.nodeIsPlaceHolder(node.nodeName())) {
                val funcAndContext = createFuncAndContext(node.opName(),
                    irGraph,opMappingRegistry,
                    sd,node.nodeName(),
                    dynamicVariables)
                nodeNameToFuncContext[node.nodeName()] = funcAndContext
            }
        }

        //get an updated set of number of nodes
        nNodes = irGraph.nodeList().size
        //Setup initial inputs
        for (i in 0 until nNodes) {
            val nd = irGraph.nodeList()[i]
            val name = nd.nodeName()
            if(name.isEmpty()) {
                println("Skipping node $i due to empty name.")
                continue
            }
            val op = nd.opName()
            val numInputs = nd.numInputs()
            val numOutputs = nd.numOutputs()
            Preconditions.checkState(name.isNotEmpty(), "Node name was empty!")
            if (irGraph.isConstantOpName(op)|| numInputs == 0) {
                availableToAdd.add(nd)
                availableToAddSet.add(name)
                logger.debug {"Added $name" }
            } else {
                remainingNodes[name] = nd


                for (inputIdx in 0 until numInputs) {
                    var inOpName = stripVarSuffix(stripControl(nd.inputAt(inputIdx)))
                    if (!nodeInputTo.containsKey(inOpName)) {
                        nodeInputTo[inOpName!!] = ListOrderedSet()
                    }
                    //don't add the same name twice, we risk repeating additions above
                    if(!nodeInputTo[inOpName]!!.contains(name))
                        nodeInputTo[inOpName]!!.add(name)
                }

                if(irGraph.addGraphOutputsAsProcessingNodes()) {
                    //add outputs or current nodes to available to add
                    //queue to ensure processing happens
                    //some frameworks have independent output names of actual nodes
                    //in this case, nodes should be added
                    for(outputIdx in 0 until numOutputs) {
                        var outOpName = stripVarSuffix(stripControl(nd.outputAt(outputIdx)))
                        if(irGraph.hasNode(outOpName) && !irGraph.isConstant(outOpName)) {
                            availableToAdd.add(irGraph.irNodeByName(outOpName))
                            availableToAddSet.add(outOpName)
                        } else {
                            //no node for output name, avoid duplicates being added to the processing
                            //queue
                            if(!availableToAddSet.contains(nd.nodeName())) {
                                availableToAdd.add(nd)
                                availableToAddSet.add(nd.nodeName())
                            }
                        }
                    }
                }
            }
        }


        val mergeOpsPostProcess: MutableMap<String, String> = HashMap()
        //Go through ops in order, and add to the graph
        val constControlDeps: MutableMap<String, List<String>> = HashMap() //Key: constant name. Value: control dependencies


        while (!availableToAdd.isEmpty()) {
            val nd = availableToAdd.remove()
            val name = nd.nodeName()
            if(name.isEmpty()) {
                continue
            }
            availableToAddSet.remove(name)
            logger.debug {"Removed $name" }
            val opName = nd.opName()
            val importInfoForNode = importInfo[name]
            val opMappingProcess = OpRegistryHolder.lookupOpMappingProcess<
                    GRAPH_TYPE,
                    NODE_TYPE,
                    OP_DEF_TYPE,
                    TENSOR_TYPE,
                    DATA_TYPE,
                    ATTR_DEF_TYPE,
                    ATTR_VALUE_TYPE>(inputFrameworkOpName = opName, inputFrameworkName = irGraph.frameworkName())

            val funcContextResult = nodeNameToFuncContext[nd.nodeName()]
            /*
                Normal ops. Process in the following order:
                1. Create the op instance
                2. Add op to graph
                3. Import from TF (to set attributes)
                4. Calculate output dtypes
                5. Create and add output variables to graph
                 */

            var df = funcContextResult?.dfInstance ?: Identity()


            val mappingContext = funcContextResult?.mappingContext
            val nd4jOpName = df.opName()



            logger.debug {"Adding operation to graph: $opName (name=$name)"}
            opsAdded.add("$opName,$name")
            var skipCase = false
            val rawAttrMap = HashMap<String, ATTR_VALUE_TYPE>()
            nd.attributeMap().forEach { (name, def) ->
                rawAttrMap[name] = def.internalAttributeValue()
            }


            if (opFilter != null && opFilter.skipOp(
                    nd.internalValue(),
                    sd,rawAttrMap, irGraph.internalValue())) {
                logger.debug {"Skipping op $name of type $opName due to op filter" }
                //Don't continue at this point - we still need to process what this feeds into...
                skipCase = true
            } else {
                if (importOverride == null || !importOverride.containsKey(name)) {
                    //Standard case
                    //note, ordering matters here for onnx
                    if (irGraph.nodeIsPlaceHolder(nd.nodeName())) {
                        logger.debug {"Adding placeholder ${nd.nodeName()}" }
                        if(!sd.hasVariable(nd.nodeName())) {
                            var shape = irGraph.shapeOfInput(nd.nodeName())
                            val dt = irGraph.dataTypeForVariable(nd.nodeName()).nd4jDataType()
                            if(shape != null)
                                sd.placeHolder(name, dt, *shape)
                            else
                                sd.placeHolder(name, dt)
                        } else {
                            val sdVar = sd.getVariable(nd.nodeName())
                            sdVar.variableType = VariableType.PLACEHOLDER
                            sdVar.creator = df
                            val dt = irGraph.dataTypeForVariable(nd.nodeName()).nd4jDataType()
                            sdVar.setDataType(dt)
                            if(sdVar.arr == null && dynamicVariables.containsKey(nd.nodeName())) {
                                //ensure we set the array to the proper data type
                                val castedArr = irGraph.convertToNDArray(dynamicVariables[nd.nodeName()]!!).castTo(dt)
                                sd.associateArrayWithVariable(castedArr,sdVar)
                                dynamicVariables[nd.nodeName()] = irGraph.convertToTensor(castedArr,nd.nodeName())
                                sd.setEagerArrForVarName(nd.nodeName(),castedArr)

                            }


                        }

                    }
                    else if (irGraph.isConstant(opName)) {
                        if(!sd.hasVariable(nd.nodeName())) {
                            logger.debug {"Adding constant ${nd.nodeName()}" }
                            //Get array, create a constant
                            val arr = irGraph.getConstantArrayForName(name)
                            sd.constant(name, arr)
                            logger.debug {"Added constant for node name ${nd.nodeName()} with shape ${arr.shapeInfoToString()}" }
                            val inputCount = nd.numInputs()
                            if (inputCount > 0) {
                                //Very likely control dependency. i.e., "we must execute op X before the constant is really available to be used"
                                val l: MutableList<String> = ArrayList(inputCount)
                                for (i in 0 until inputCount) {
                                    val n = nd.inputAt(i)
                                    check(isControlDep(n)) { "Found non-control dependency input \"$n\" for constant \"$name\"" }
                                    val n2 = stripControl(n)
                                    l.add(n2)
                                }
                                constControlDeps[name] = l
                            }
                        } else {
                            val varToGet = sd.getVariable(nd.nodeName())
                            varToGet.variableType = VariableType.CONSTANT
                            varToGet.creator = df
                            if(sd.getVariable(nd.nodeName()).arr == null) {
                                val arr = irGraph.getConstantArrayForName(name)
                                varToGet.setArray(arr)
                                varToGet.setShape(*arr.shape())
                            }

                        }
                    }  else if(irGraph.isVariable(nd.nodeName()) && !sd.hasVariable(nd.nodeName())) {
                        var shape = irGraph.shapeOfInput(nd.nodeName())
                        val dt = irGraph.dataTypeForVariable(nd.nodeName()).nd4jDataType()
                        if(shape != null)
                            sd.`var`(name, dt, *shape)
                        else
                            sd.`var`(name, dt,-1)
                    }
                    else if(nodeNameToFuncContext.containsKey(nd.nodeName())) {

                        //Process inputs
                        var controlDeps: MutableList<String?>? = null
                        val numInputs = nd.numInputs()
                        val inNames: MutableList<String> = ArrayList(numInputs)

                        for (i in 0 until numInputs) {
                            //use input name if it exists and matches, otherwise if the input names do not map 1 to 1 for import
                            //use samediff to generate a unique name
                            val origInName = nd.inputAt(i)
                            var inName = stripControl(origInName)
                            if (inName.endsWith(":0")) {
                                //Strip ":0" suffix. Some ops can depend on placeholders, like "image_tensor:0" but in SameDiff this is a variable called "image_tensor"
                                inName = inName.substring(0, inName.length - 2)
                            }
                            val isControlDep = isControlDep(origInName)
                            if (isControlDep) {
                                if (controlDeps == null) controlDeps = ArrayList()
                                controlDeps.add(inName)
                            }
                            if (!isControlDep) {
                                inNames.add(inName)
                            }

                            //Update Variable.inputsForOp for all variables that feed into this op
                            // Such variables must have already been created, given we process in order
                            //declare empty variable for anything that's an input > 0
                            if(!sd.hasVariable(inName) && inName.contains(':')) {
                                val knownBaseName = stripVarSuffix(inName)
                                if(!sd.hasVariable(knownBaseName)) {
                                    throw IllegalArgumentException("No variable name found for $knownBaseName")
                                }
                                else {
                                    val knownBaseVar = sd.getVariable(stripVarSuffix(inName))
                                    sd.`var`(
                                        SDVariable(
                                            inName,
                                            VariableType.ARRAY,
                                            sd,
                                            knownBaseVar.shape,
                                            knownBaseVar.dataType()
                                        )
                                    )

                                }
                            }

                            //auto declare variables if they don't exist, avoid constants. Pull constants out
                            //from the graph and initialize them if an input name appears before a mention of a constant.
                            //This can happen in certain frameworks. Sometimes frameworks will have auto sorted
                            //DAGS, this may not be true for all situations though.
                            //note, we only want variables being auto declared if they are actually inputs or outputs not only nodes
                            if(!isControlDep && !sd.hasVariable(inName) && !irGraph.hasConstantInitializer(inName) && irGraph.isInputOrOutput(inName)) {
                                val otherInputs = nd.inputs().filter { input -> sd.hasVariable(input) }
                                var dataType = DataType.FLOAT
                                //guess input from other data types
                                if(otherInputs.isNotEmpty()) {
                                    dataType = sd.getVariable(otherInputs[0]).dataType()
                                }
                                sd.`var`(
                                    SDVariable(
                                        inName,
                                        VariableType.ARRAY,
                                        sd,
                                        null,
                                        dataType
                                    )
                                )
                            } else if(!isControlDep && !sd.hasVariable(inName) && irGraph.hasConstantInitializer(inName)) {
                                val const = irGraph.getConstantArrayForName(inName)
                                sd.constant(inName,const)
                            } else if(!isControlDep && !sd.hasVariable(inName)) {
                                throw IllegalStateException("Input variable at index $i named $inName of node $name was not assigned to any variable")
                            }

                            val v = sd.variables[inName]
                            if (v == null && df is Merge) {
                                //Edge case for import - we allow merge ops to be added before both inputs are available
                                //This is to break the cycles in loops, otherwise we can't process anything in order
                                mergeOpsPostProcess[df.getOwnName()] = inName
                                continue
                            }

                            if (v != null && !isControlDep && (v!!.inputsForOp == null || !v.inputsForOp.contains(name))) {
                                //May already be present - for example, add(x,x)
                                if (v.inputsForOp == null) v.inputsForOp = ArrayList()
                                v.inputsForOp.add(name)
                            } else if (v != null && isControlDep) {
                                if (v!!.controlDepsForOp == null) v.controlDepsForOp = ArrayList()
                                if (!v.controlDepsForOp.contains(name)) {
                                    v.controlDepsForOp.add(name)
                                }
                            }


                        }

                        //ensure every function has an op name set (mainly for debugging)
                        if(df is DynamicCustomOp) {
                            val opField = DynamicCustomOp::class.java.getDeclaredField("opName")
                            opField.isAccessible = true
                            ReflectionUtils.setField(opField,df,nd4jOpName)
                        }


                        //Create SameDiffOp instance and add to graph
                        val op = SameDiffOp.builder()
                            .name(name)
                            .op(df)
                            .controlDeps(controlDeps)
                            .build()
                        //take only up to the inputs that are specified in the node/
                        //this is for cases where node inputs is > intended number for ops
                        //a common example is when ops convert input ndarrays to integers or float inputs
                        val resolvedArgInputs = importInfo[name]!!.second.argDescriptorList.filter {input -> input.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR}
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
                                                    sd.`var`(op.inputsToOp[arg.argIndex],ndarrayFromNameSpaceTensor(arg.inputValue))
                                                } else {
                                                    throw java.lang.IllegalArgumentException("No argument value found for op ${op.name} for value arg with name ${arg.name}")
                                                }
                                            }
                                        }
                                    }  else
                                        op.inputsToOp = inNames

                                    //clear out inputs for variables as well to reflect the actual graph structure
                                    //NO OP NOTE: we make an exception for no op mappings. No op mappings are a potential
                                    //signal that we are using  pre hook rules as substitutions for operations
                                    //without a mapping process. This can happen when we don't have an exact mapping
                                    //for an op and need a way of substituting the op with an equivalent set of samediff
                                    //op calls. Specifying no nop as a way of handling mapping processes allows a special
                                    //sentinel value  that , in combination with pre hook rules, can be used
                                    //to substitute ops when they may not otherwise be supported.
                                    //The reason we can't take away the inputs is the user may specify the inputs
                                    //to the op and the hook rule may need those inputs to use as a base for calculations.
                                    if(numInputsToTake < numInputs && op.op.opName() != "noop") {
                                        for(i in numInputsToTake until numInputs) {
                                            if(sd.hasVariable(nd.inputAt(i))) {
                                                val currInputVar = sd.variables[nd.inputAt(i)]!!
                                                currInputVar.inputsForOp.remove(op.name)
                                            }
                                        }
                                    }

                                }
                                MapperNamespace.VariableResolutionType.ERROR_ON_NOT_EQUAL -> {
                                    throw java.lang.IllegalStateException("Number of variable names for node ${mappingContext!!.nodeName()} not exact equal to number of inputs resolved from nd4j op descriptor which was ${resolvedArgInputs.size}")
                                }
                            }

                            //we want the default names used for no op or other situations
                        } else
                            op.inputsToOp = inNames


                        //cache attributes just in case we have any rules so we don't create the rules more than once
                        val attributes = mappingContext!!.nodeAttributesAsMap()
                        var proceedWithInit = true
                        mappingContext!!.relevantPrehookRules().forEach { rule ->
                            proceedWithInit = proceedWithInit && rule.preProcess(
                                op,
                                sd,
                                attributes,
                                importInfo[name]!!.second,
                                nd.outputs(),
                                availableToAdd.isEmpty(),
                                opMappingRegistry as OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
                                this as ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
                                dynamicVariables as Map<String,GeneratedMessageV3>
                            ).proceedWithInit
                        }

                        //add nodes/other pre processing in order for this node to work
                        if(proceedWithInit && !sd.ops.containsKey(name))
                            sd.ops[name] = op

                        if(proceedWithInit)
                            defaultRunner.initAttributes(df, sd, importInfo[name]!!)


                        //add nodes/other post processing in order for this node to work
                        mappingContext.relevantPosthookRules().forEach { rule ->
                            rule.postProcess(op, sd, attributes, importInfo[name]!!.second,nd.outputs())
                        }

                        //only add to the graph if the pre processing didn't over ride the node

                        //DType calculate for output variables (set/correct if necessary)
                        if(mappingContext.relevantPrehookRules().isEmpty()) {
                            val newInNames = sd.ops[name]!!.inputsToOp //Just in case import has modified this, like for concat case
                            val newInDtypes: MutableList<DataType> =
                                ArrayList(newInNames.size)
                            if (df is Merge) {
                                //Merge op: as noted elsewhere, we allow merge to be processed when only one of the inputs is available
                                // to break cycles for loops
                                //We know that Merge op has the restriction of the same datatype for both inputs, so we'll
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

                            //note we validate the op definition here to ensure that all ops have at least 1 output unless otherwise specified.
                            val outputDataTypes = df.calculateOutputDataTypes(newInDtypes)
                            val numOutputs = outputDataTypes.size
                            if(numInputs < 1 &&  nd4jOpName != "noop") {
                                throw IllegalStateException("Op $nd4jOpName does not have any outputs!")
                            }

                            //logger.debug {"Out dtypes size ${outDTypes.size} and numOutputs $numOutputs")
                            val outSDVars = arrayOfNulls<SDVariable>(numOutputs)
                            val outVars = arrayOfNulls<Variable>(numOutputs)
                            val outNames: MutableList<String> = ArrayList(numOutputs)

                            //Create output variables and add to graph
                            for (i in 0 until numOutputs) {
                                val dt = outputDataTypes[i]
                                val varName = nd.outputAt(i)

                                outSDVars[i] = if(sd.hasVariable(varName)) sd.getVariable(varName) else sd.`var`(varName, VariableType.ARRAY, null, dt)
                                outNames.add(varName)
                                if(sd.variables.containsKey(varName)) {
                                    outVars[i] = sd.variables[varName]
                                    if(outVars[i]!!.variable == null)
                                        outVars[i]!!.variable = outSDVars[i]
                                    if(outVars[i]!!.outputOfOp == null) {
                                        outVars[i]!!.outputOfOp = name
                                    }
                                }
                                else {
                                    outVars[i] = Variable.builder()
                                        .name(varName)
                                        .variable(outSDVars[i])
                                        .inputsForOp(null) //This is updated incrementally as other ops are added
                                        .controlDepsForOp(null) //Control deps are handled later
                                        .controlDepsForVar(null)
                                        .outputOfOp(name)
                                        .build()
                                    sd.variables[varName] = outVars[i]
                                }
                                logger.debug {"Added variable to graph: $varName (output of op $name)" }
                                variablesAdded.add("$varName,$name")
                            }

                            sd.ops[name]!!.outputsOfOp = outNames

                            //don't run computeArrays if graph contains control flow, too many edge cases
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
                            logger.debug {"Imported op: $opName (name=$name)" }
                            opsImported.add("$opName,$name")

                        }


                    }
                    else {
                        logger.debug {"Node ${nd.nodeName()} not found in import context, skipping!" }
                    }
                } else {


                    val dfInstance = if( DifferentialFunctionClassHolder.getInstance()
                            .hasName(opName)) DifferentialFunctionClassHolder.getInstance().getInstance(opName)
                    else DynamicCustomOp.builder(opName).build()
                    Preconditions.checkState(
                        dfInstance != null,
                        "Could not find class for ${opMappingProcess.opName()}",
                        opName
                    )
                    var df: DifferentialFunction
                    df = try {
                        dfInstance.javaClass.newInstance()
                    } catch (t: Throwable) {
                        //Should never happen because function was already created via no-arg constructor earlier
                        throw RuntimeException(t)
                    }

                    df.sameDiff = sd
                    df.ownName = name



                    //Import override case
                    val o = importOverride[name]
                    logger.debug {"Importing op $opName using override $importOverride" }

                    //First, get inputs:
                    val inputs: MutableList<SDVariable> = ArrayList()
                    var controlDeps: MutableList<SDVariable?>? = null
                    val nd4jOpName = opMappingRegistry.lookupOpMappingProcess(opName).opName()
                    val opDescriptor = opMappingRegistry.lookupNd4jOpDef(nd4jOpName)
                    val opInputs = opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
                    val numInputs = opInputs.size


                    for (i in 0 until numInputs) {
                        val inName = nodeInputTo[nd.nodeName()]!![i]!!
                        val controlDep = isControlDep(inName)
                        val v = sd.getVariable(name)
                        if (controlDep) {
                            if (controlDeps == null) controlDeps = ArrayList()
                            controlDeps.add(v)
                        } else {
                            inputs.add(v)
                        }

                        o!!.initAttributes(df, sd, importInfo[nd.nodeName()]!!)
                    }
                }
            }


            //Now that we have just added an op (or variable) - check what this feeds into, and see what we can now process
            // as a result
            if (nodeInputTo.containsKey(name)) {
                val set: ListOrderedSet<String>? = nodeInputTo[name]
                for (nextOp in set!!) {
                    val nextOpDef = remainingNodes[nextOp]

                    if (nextOpDef == null) {
                        val opSet = setOf("noop","assert","const","merge")
                        if (sd.ops.containsKey(nextOp) || opSet.contains(importInfoForNode!!.first.nd4jOpName())) {
                            //Already processed this.
                            //Almost certainly the close of a loop - like NextIteration -> Merge case
                            continue
                        }
                        throw IllegalStateException("Could not find op definition for op to import: $nextOp")
                    }

                    val nInNext = nextOpDef.numInputs()
                    var allAlreadyInGraph = true
                    var nonControlSeenCount = 0

                    for (i in 0 until nInNext) {
                        val s = nextOpDef.inputAt(i)
                        var inName = stripControl((nextOpDef.inputAt(i)))
                        if (inName.endsWith(":0")) {
                            //Strip ":0" suffix. Some ops can depend on placeholders, like "image_tensor:0" but in SameDiff this is a variable called "image_tensor"
                            inName = inName.substring(0, inName.length - 2)
                        }

//                        log.info("Input: {}, {}", s, inName);
                        //note on initializers, sometimes ops mentions pre initialized constants
                        //that haven't been seen by import yet. In this case, we need to allow the
                        //op to be added, otherwise no further import can happen
                        if (!sd.hasVariable(inName) && !skipCase && !irGraph.hasConstantInitializer(inName) && !irGraph.hasConstantInitializer(inName)) {
//                            log.info("Not found: {} for op {}", inName, nextOpDef.getName());
                            allAlreadyInGraph = false
                            break
                        } else if (!isControlDep(s)) {
                            nonControlSeenCount++
                        }
                    }

                    //Merge ops are an edge case. We'll allow these to be executed with just ONE input, to break
                    // the cycle in loops. In loops, generally we have (Enter, NextIteration) -> Merge, which
                    // of course can't be done if we strictly require all inputs to be available
                    val mergeCase = nonControlSeenCount > 0 && "Merge" == nextOpDef.opName()
                    if (allAlreadyInGraph || mergeCase) {
                        //Can process this op, add it to the queue for processing
                        if (!availableToAddSet.contains(nextOp)) {
                            //Avoid processing same op multiple times, for repeated inputs to one op, etc
                            availableToAdd.add(nextOpDef)
                            logger.debug {"Added ${nextOpDef.nodeName()}" }
                            availableToAddSet.add(nextOp)
                            logger.debug {"Added to processing queue: ${nextOpDef.opName()} (name=$nextOp)" }
                        }
                    }
                }
            }

            //Finally, remove the just processed op from remainingNodes map:
            remainingNodes.remove(name)
            opsRemoved.add(name)
        }

        //Post process the control dependencies, if any (done after because dependencies may not exist when imported)
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

        //Post process the merge ops - all we are missing is a Variable.getInputsForOp().add(mergeOpName);
        for ((key, value) in mergeOpsPostProcess) {
            val v = sd.variables[value]
            if(v != null) {
                if ( v!!.inputsForOp == null) v.inputsForOp = ArrayList()
                v.inputsForOp.add(key)
            }

        }


        logger.debug {"Variables added $variablesAdded"}
        logger.debug {"Ops imported $opsImported"}
        logger.debug {"Ops added $opsAdded"}
        logger.debug {"Ops removed $opsRemoved"}


        Preconditions.checkState(
            remainingNodes.isEmpty(),
            "%s Unprocessed nodes: %s",
            remainingNodes.size,
            remainingNodes.keys
        )

        val opByOutputName = HashMap<String,MutableList<SameDiffOp>>()
        sd.ops.forEach { (opName, op) ->
            val opOutput = op.outputsOfOp[0]
            if(!opByOutputName.containsKey(opOutput)) {
                opByOutputName[opOutput] = ArrayList()
            }

            val list = opByOutputName[opOutput]!!
            list.add(op)
        }

        println(sd.summary())


        return sd
    }

    private fun renameOp(
        secondOp: SameDiffOp,
        firstOp: SameDiffOp,
        sd: SameDiff
    ) {
        val realOp = secondOp.op
        val realName = firstOp.op.ownName
        val oldOp = firstOp.op
        val realControlDeps = secondOp.controlDeps
        val realVarControlDeps = secondOp.varControlDeps
        val realInputs = secondOp.inputsToOp
        val oldName = secondOp.op.ownName
        firstOp.op = realOp
        //firstOp.inputsToOp = realInputs
        firstOp.op.ownName = realName
        firstOp.controlDeps = realControlDeps
        firstOp.varControlDeps = realVarControlDeps
        sd.ops.forEach { opName, op ->
            if (op.inputsToOp != null && op.inputsToOp.contains(oldName)) {
                op.inputsToOp[op.inputsToOp.indexOf(oldName)] = realName
            }

            if (op.controlDepFor != null && op.controlDepFor.contains(oldName)) {
                op.controlDepFor[op.controlDepFor.indexOf(oldName)] = realName
            }

            if (op.controlDeps != null && op.controlDeps.contains(oldName)) {
                op.controlDeps[op.controlDeps.indexOf(oldName)] = realName
            }
        }
        sd.ops.remove(secondOp.name)
    }
}

