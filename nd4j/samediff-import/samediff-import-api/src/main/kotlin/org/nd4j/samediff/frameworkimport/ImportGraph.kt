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
import org.apache.commons.io.FileUtils
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
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge
import org.nd4j.linalg.api.ops.impl.shape.Concat
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.registry.OpRegistryHolder
import org.nd4j.samediff.frameworkimport.runner.DefaultImportRunner
import org.nd4j.samediff.frameworkimport.runner.ImportRunner
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.io.File
import java.lang.IllegalArgumentException
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.HashSet


open class ImportGraph <GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTR_DEF_TYPE : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {


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
        var df: DifferentialFunction
        df = try {
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

        val availableToAddSet = HashSet<String>() //TODO maybe unnecessary?
        val availableToAdd: Queue<IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>> = LinkedList()
        val remainingNodes: MutableMap<String, IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>> =
            HashMap() //All other nodes, not in availableToAdd
        val nodeInputTo: MutableMap<String, ListOrderedSet<String>> =
            HashMap() // For op x -> y, x is key, y is value. Note that these are OP names not VARIABLE names
        var nNodes: Int
        val importInfo = irGraph.importInfoForEachNode(dynamicVariables = dynamicVariables)
        //First, add any constants, placeholders, and zero-input ops
        val sd = SameDiff.create()
        val defaultRunner =
            DefaultImportRunner<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>()

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

        for (i in 0 until nNodes) {
            val nd = irGraph.nodeList()[i]

            val op = nd.opName()
            val numInputs = nd.numInputs()
            val name = nd.nodeName()
            Preconditions.checkState(name.isNotEmpty(), "Node name was empty!")
            if (irGraph.isConstantOpName(op)|| numInputs == 0) {
                availableToAdd.add(nd)
                availableToAddSet.add(name)
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

            }
        }


        val mergeOpsPostProcess: MutableMap<String, String> = HashMap()
        //Go through ops in order, and add to the graph
        val constControlDeps: MutableMap<String, List<String>> = HashMap() //Key: constant name. Value: control dependencies
        while (!availableToAdd.isEmpty()) {
            val nd = availableToAdd.remove()
            val name = nd.nodeName()
            val opName = nd.opName()
            val importInfoForNode = importInfo[name]

            availableToAddSet.remove(name)
            println("Adding operation to graph: $opName (name=$name)")
            opsAdded.add(opName + "," + name)
            var skipCase = false
            val rawAttrMap = HashMap<String, ATTR_VALUE_TYPE>()
            nd.attributeMap().forEach { (name, def) ->
                rawAttrMap[name] = def.internalAttributeValue()
            }


            if (opFilter != null && opFilter.skipOp(
                    nd.internalValue(),
                    sd,rawAttrMap, irGraph.internalValue())) {
                println("Skipping op $name of type $opName due to op filter")
                //Don't continue at this point - we still need to process what this feeds into...
                skipCase = true
            } else {
                if (importOverride == null || !importOverride.containsKey(name)) {
                    //Standard case
                    //note, ordering matters here for onnx
                    if (irGraph.nodeIsPlaceHolder(nd.nodeName()) && !sd.hasVariable(nd.nodeName())) {
                        println("Adding placeholder ${nd.nodeName()}")
                        var shape = irGraph.shapeOfInput(nd.nodeName())
                        val dt = irGraph.dataTypeForVariable(nd.nodeName()).nd4jDataType()
                        if(shape != null)
                            sd.placeHolder(name, dt, *shape)
                        else
                            sd.placeHolder(name, dt)
                    }
                    else if (irGraph.isConstant(opName) && !sd.hasVariable(name)) {
                        println("Adding constant ${nd.nodeName()}")
                        //Get array, create a constant
                        val arr = irGraph.getConstantArrayForName(name)
                        sd.constant(name, arr)
                        println("Added constant for node name ${nd.nodeName()} with shape ${arr.shapeInfoToString()}")
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
                    }  else if(irGraph.isVariable(nd.nodeName()) && !sd.hasVariable(nd.nodeName())) {
                        var shape = irGraph.shapeOfInput(nd.nodeName())
                        val dt = irGraph.dataTypeForVariable(nd.nodeName()).nd4jDataType()
                        if(shape != null)
                            sd.`var`(name, dt, *shape)
                        else
                            sd.`var`(name, dt,-1)
                    }
                    else if(nodeNameToFuncContext.containsKey(nd.nodeName())) {
                        val funcContextResult = nodeNameToFuncContext[nd.nodeName()]!!
                        /*
                            Normal ops. Process in the following order:
                            1. Create the op instance
                            2. Add op to graph
                            3. Import from TF (to set attributes)
                            4. Calculate output dtypes
                            5. Create and add output variables to graph
                             */

                        var df = funcContextResult.dfInstance
                        val mappingContext = funcContextResult.mappingContext
                        val tensorInputMappings = funcContextResult.tensorInputMappings
                        val nd4jOpName = df.opName()
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
                                    throw IllegalArgumentException("No variable name found for $inName")
                                } else {
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

                            val v = sd.variables[inName]
                            if (v == null && df is Merge) {
                                //Edge case for import - we allow merge ops to be added before both inputs are available
                                //This is to break the cycles in loops, otherwise we can't process anything in order
                                mergeOpsPostProcess[df.getOwnName()] = inName
                                continue
                            }

                            if (!isControlDep && (v!!.inputsForOp == null || !v.inputsForOp.contains(name))) {
                                //May already be present - for example, add(x,x)
                                if (v.inputsForOp == null) v.inputsForOp = java.util.ArrayList()
                                v.inputsForOp.add(name)
                            } else if (isControlDep) {
                                if (v!!.controlDepsForOp == null) v.controlDepsForOp = java.util.ArrayList()
                                if (!v.controlDepsForOp.contains(name)) {
                                    v.controlDepsForOp.add(name)
                                }
                            }


                        }

                        val inputNames = nd.nd4jInputs(tensorInputMappings)
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
                        val numInputsToTake = importInfo[name]!!.second.argDescriptorList.filter { input -> input.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
                            .size
                        op.inputsToOp = inNames.subList(0,numInputsToTake)

                        //add nodes/other pre processing in order for this node to work
                        sd.ops[name] = op
                        //clear out inputs for variables as well to reflect the actual graph structure
                        if(numInputsToTake < numInputs) {
                            for(i in numInputsToTake until numInputs) {
                                if(sd.hasVariable(nd.inputAt(i))) {
                                    val currInputVar = sd.variables[nd.inputAt(i)]!!
                                    currInputVar.inputsForOp.remove(op.name)
                                }
                            }
                        }

                        //cache attributes just in case we have any rules so we don't create the rules more than once
                        val attributes = mappingContext.nodeAttributesAsMap()
                        mappingContext.relevantPrehookRules().forEach { rule ->
                            rule.preProcess(op, sd,attributes)
                        }

                        defaultRunner.initAttributes(df, sd, importInfo[name]!!)


                        //add nodes/other post processing in order for this node to work
                        mappingContext.relevantPosthookRules().forEach { rule ->
                            rule.postProcess(op, sd,attributes)
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
                            } else if(df is Concat) {
                                //note we use the nd4j data types here so we only have input data types indexed by the actual
                                //output from nd4j. A common scenario import is dimensions being converted to ints
                                //Dimensions are converted from inputs in the input framework to plain integers elsewhere.
                                //This lets the import process dictate the actual ordering of the data types.
                                for (s in inputNames) {
                                    val v = sd.getVariable(s)
                                    newInDtypes.add(v.dataType())
                                }

                                op.inputsToOp = inputNames
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

                            //println("Out dtypes size ${outDTypes.size} and numOutputs $numOutputs")
                            val outSDVars = arrayOfNulls<SDVariable>(numOutputs)
                            val outVars = arrayOfNulls<Variable>(numOutputs)
                            val outNames: MutableList<String> = ArrayList(numOutputs)

                            //Create output variables and add to graph
                            for (i in 0 until numOutputs) {
                                val dt = outputDataTypes[i]
                                val varName = name + if (i == 0) "" else ":$i"
                                //TODO: handle variadic type in kotlin
                                /**
                                 * TODO: handle data type import
                                 */
                                outSDVars[i] = if(sd.hasVariable(varName)) sd.getVariable(varName) else sd.`var`(varName, VariableType.ARRAY, null, dt)
                                outNames.add(varName)
                                outVars[i] = Variable.builder()
                                    .name(varName)
                                    .variable(outSDVars[i])
                                    .inputsForOp(null) //This is updated incrementally as other ops are added
                                    .controlDepsForOp(null) //Control deps are handled later
                                    .controlDepsForVar(null)
                                    .outputOfOp(name)
                                    .build()
                                sd.variables[varName] = outVars[i]
                                println("Added variable to graph: $varName (output of op $name)")
                                variablesAdded.add(varName + "," + name)
                            }

                            sd.ops[name]!!.outputsOfOp = outNames
                            println("Imported op: $opName (name=$name)")
                            opsImported.add(opName + "," + name)

                        }


                    }
                    else {
                        println("Node ${nd.nodeName()} not found in import context, skipping!")
                    }
                } else {

                    val opMappingProcess = OpRegistryHolder.lookupOpMappingProcess<
                            GRAPH_TYPE,
                            NODE_TYPE,
                            OP_DEF_TYPE,
                            TENSOR_TYPE,
                            DATA_TYPE,
                            ATTR_DEF_TYPE,
                            ATTR_VALUE_TYPE>(inputFrameworkOpName = opName, inputFrameworkName = irGraph.frameworkName())



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

                    val opDefLookup = opMappingRegistry.lookupInputFrameworkOpDef(opName) as OP_DEF_TYPE


                    //Import override case
                    val o = importOverride[name]
                    println("Importing op $opName using override $importOverride")

                    //First, get inputs:
                    val inputs: MutableList<SDVariable> = java.util.ArrayList()
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
                            if (controlDeps == null) controlDeps = java.util.ArrayList()
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
                        val opSet = setOf("noop","assert","while","identity","const","merge")
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
                        var inName = stripControl(stripVarSuffix((nextOpDef.inputAt(i))))
                        if (inName.endsWith(":0")) {
                            //Strip ":0" suffix. Some ops can depend on placeholders, like "image_tensor:0" but in SameDiff this is a variable called "image_tensor"
                            inName = inName.substring(0, inName.length - 2)
                        }

//                        log.info("Input: {}, {}", s, inName);
                        if (!sd.hasVariable(inName) && !skipCase) {
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
                            availableToAddSet.add(nextOp)
                            println("Added to processing queue: ${nextOpDef.opName()} (name=$nextOp)")
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
                    if (sdo!!.controlDepFor == null) sdo.controlDepFor = java.util.ArrayList()
                    val l = sdo.controlDepFor
                    if (!l.contains(s)) l.add(varName)
                }
            }
        }

        //Post process the merge ops - all we are missing is a Variable.getInputsForOp().add(mergeOpName);
        for ((key, value) in mergeOpsPostProcess) {
            val v = sd.variables[value]
            if(v != null) {
                if ( v!!.inputsForOp == null) v.inputsForOp = java.util.ArrayList()
                v.inputsForOp.add(key)
            }

        }


        println("Variables added $variablesAdded")
        FileUtils.writeLines(File("variables-added-new.txt"),variablesAdded)
        println("Ops imported $opsImported")
        FileUtils.writeLines(File("ops-imported-new.txt"),opsImported)
        println("Ops added$opsAdded")
        FileUtils.writeLines(File("ops-added-new.txt"),opsAdded)
        println("Ops removed $opsRemoved")
        FileUtils.writeLines(File("ops-removed-new.txt"),opsRemoved)

        Preconditions.checkState(
            remainingNodes.isEmpty(),
            "%s Unprocessed nodes: %s",
            remainingNodes.size,
            remainingNodes.keys
        )
        return sd
    }
}

