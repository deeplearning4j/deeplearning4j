package org.nd4j.codegen.ir

import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.autodiff.samediff.internal.Variable
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.codegen.ir.tensorflow.isControlDep
import org.nd4j.codegen.ir.tensorflow.stripControl
import org.nd4j.codegen.ir.tensorflow.stripVarSuffix
import org.nd4j.common.base.Preconditions
import org.nd4j.imports.converters.DifferentialFunctionClassHolder
import org.nd4j.imports.graphmapper.OpImportFilter
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge
import org.nd4j.linalg.api.ops.impl.shape.Concat
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.lang.IllegalArgumentException
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.HashSet


class ImportGraph <GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTR_DEF_TYPE : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {
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
                        dynamicVariables: Map<String, TENSOR_TYPE> = emptyMap(),
                        opMappingRegistry: OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE>
    ): SameDiff {

        /*
            First, build an in-memory representation of the graph that allows us to build the graph incrementally
            If we can build the graph incrementally, we can make sure that the added variables are set up with the correct
            datatype and (once implemented) greedy shape inference
             */
        val availableToAddSet = HashSet<String>() //TODO maybe unnecessary?
        val availableToAdd: Queue<IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>> = LinkedList()
        val remainingNodes: MutableMap<String, IRNode<NODE_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>> =
            HashMap() //All other nodes, not in availableToAdd
        val nodeInputTo: MutableMap<String, MutableList<String>> =
            HashMap() // For op x -> y, x is key, y is value. Note that these are OP names not VARIABLE names
        val nNodes = irGraph.nodeList().size
        val importInfo = irGraph.importInfoForEachNode(dynamicVariables = dynamicVariables)
        //First, add any constants, placeholders, and zero-input ops
        val sd = SameDiff.create()
        irGraph.nodeList().forEach { node ->
            val importInfoForNode = importInfo[node.nodeName()]!!
            val numInputs = node.numInputs()
            val nodeInputs = ArrayList<String>()
            val name = node.nodeName()

            for(inputIdx in 0 until numInputs) {
                var inOpName = stripVarSuffix(stripControl(node.inputAt(inputIdx)))
                nodeInputs.add(inOpName)
                if (!nodeInputTo.containsKey(inOpName)) {
                    nodeInputTo[inOpName!!] = ArrayList()
                }

                nodeInputTo[inOpName]!!.add(name)
            }

            val inputs = importInfoForNode.second.argDescriptorList.filter { input -> input.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
            if(numInputs < inputs.size) {
                for(i in numInputs until inputs.size) {
                    val newName = name + "-" + inputs[i].name
                    nodeInputTo[newName!!] = ArrayList()
                    nodeInputTo[newName]!!.add(name)
                    sd.constant(newName, ndarrayFromNameSpaceTensor(inputs[i].inputValue))
                }

            }


        }

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
                    var inOpName = stripControl(nd.inputAt(inputIdx))
                    if (!nodeInputTo.containsKey(inOpName)) {
                        nodeInputTo[inOpName!!] = ArrayList()
                    }
                    nodeInputTo[inOpName]!!.add(name)
                }

            }
        }


        val mergeOpsPostProcess: MutableMap<String, String> = HashMap()
        val defaultRunner =
            DefaultImportRunner<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>()
        //Go through ops in order, and add to the graph
        val constControlDeps: MutableMap<String, List<String>> = HashMap() //Key: constant name. Value: control dependencies
        while (!availableToAdd.isEmpty()) {
            val nd = availableToAdd.remove()
            val name = nd.nodeName()
            val opName = nd.opName()
            val importInfoForNode = importInfo[name]

            availableToAddSet.remove(name)
            println("Adding operation to graph: $opName (name=$name)")
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
                    if (irGraph.nodeIsPlaceHolder(nd.nodeName())) {
                        var shape = irGraph.shapeOfInput(nd.nodeName())


                        val dt = irGraph.dataTypeForVariable(nd.nodeName()).nd4jDataType()
                        if(shape != null)
                            sd.placeHolder(name, dt, *shape)
                        else
                            sd.placeHolder(name, dt)
                    }
                    else if (irGraph.isConstant(opName)) {
                        //Get array, create a constant
                        val tfTensor = nd.getAttribute("value").tensorValue()
                        val arr = tfTensor.toNd4jNDArray()
                        sd.constant(name, arr)
                        val inputCount = nd.numInputs()
                        if (inputCount > 0) {
                            //Very likely control dependency. i.e., "we must execute op X before the constant is really available to be used"
                            val l: MutableList<String> = java.util.ArrayList(inputCount)
                            for (i in 0 until inputCount) {
                                val n = nd.inputAt(i)
                                check(isControlDep(n)) { "Found non-control dependency input \"$n\" for constant \"$name\"" }
                                val n2 = stripControl(n)
                                l.add(n2)
                            }
                            constControlDeps[name] = l
                        }
                    }  else if(opName.equals("Variable") || opName.equals("VariableV2")) {
                        var shape = irGraph.shapeOfInput(nd.nodeName())


                        val dt = irGraph.dataTypeForVariable(nd.nodeName()).nd4jDataType()
                        if(shape != null)
                            sd.`var`(name, dt, *shape)
                        else
                            sd.`var`(name, dt,-1)
                    }
                    else {
                        /*
                            Normal ops. Process in the following order:
                            1. Create the op instance
                            2. Add op to graph
                            3. Import from TF (to set attributes)
                            4. Calculate output dtypes
                            5. Create and add output variables to graph

                            Note: one constraint on this order is that some ops import modify the graph structure.
                            Notable example: concat op - it removes the axis op and converts the value to an iArg
                            https://github.com/eclipse/deeplearning4j/issues/8285
                             */

                        val opMappingProcess = OpRegistryHolder.lookupOpMappingProcess<
                                GRAPH_TYPE,
                                NODE_TYPE,
                                OP_DEF_TYPE,
                                TENSOR_TYPE,
                                DATA_TYPE,
                                ATTR_DEF_TYPE,
                                ATTR_VALUE_TYPE>(
                            inputFrameworkOpName = opName,
                            inputFrameworkName = irGraph.frameworkName()
                        )


                        val nd4jOpName = opMappingRegistry.lookupOpMappingProcess(opName).opName()

                        val dfInstance = if( DifferentialFunctionClassHolder.getInstance()
                                .hasName(nd4jOpName)) DifferentialFunctionClassHolder.getInstance().getInstance(nd4jOpName)
                        else DynamicCustomOp.builder(nd4jOpName).build()
                        Preconditions.checkState(dfInstance != null, "Could not find class for TF Ops: %s", opName)
                        var df: DifferentialFunction
                        df = try {
                            dfInstance.javaClass.newInstance()
                        } catch (t: Throwable) {
                            //Should never happen because function was already created via no-arg constructor earlier
                            throw RuntimeException(t)
                        }

                        df.sameDiff = sd
                        df.ownName = name

                        //Process inputs
                        var controlDeps: MutableList<String?>? = null
                        val numInputs = nd.numInputs()

                        /**
                         * Note that ndarrays actually need to be reordered here when input indices aren't equal to what's in the original framework.
                         * We should potentially run the import process sooner and compute the input name
                         * ordering from that instead.
                         */
                        val opDefLookup = opMappingRegistry.lookupInputFrameworkOpDef(opName)
                        val mappingContext = irGraph.createMappingContext(
                            opDef = opDefLookup,
                            node = irGraph.nodeByName(name),
                            dynamicVariables = dynamicVariables
                        )

                        val tensorInputMappings = HashMap<String, String>()
                        opMappingProcess.tensorMappingRules().forEach { tensorMappingRule ->
                            tensorInputMappings.putAll(tensorMappingRule.inputArgumentMappings())
                        }



                        val inNames: MutableList<String> = java.util.ArrayList(numInputs)

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
                                if (controlDeps == null) controlDeps = java.util.ArrayList()
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


                        val inputs = importInfoForNode!!.second.argDescriptorList.filter { input -> input.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
                        if(numInputs < inputs.size) {
                            for(i in numInputs until inputs.size) {
                                val newName = name + "-" + inputs[i].name
                                val v = sd.variables[newName]!!
                                if (v.inputsForOp == null) v.inputsForOp = java.util.ArrayList()
                                v.inputsForOp.add(newName)
                                inNames.add(newName)
                            }


                        }

                        val inputNames = nd.nd4jInputs(tensorInputMappings)


                        /**
                         * TODO: evaluate if pre/post processing is needed.
                         * May need to add new input names before and after each op.
                         * We coudl also modularize this part of the process in general.
                         */
                        //Create SameDiffOp instance and add to graph
                        val op = SameDiffOp.builder()
                            .name(name)
                            .op(df)
                            .inputsToOp(inNames) //.outputsOfOp(outNames)    //We'll set this later
                            .controlDeps(controlDeps)
                            .build()
                        sd.ops[name] = op
                        defaultRunner.initAttributes(df, irGraph.frameworkName(), mappingContext, sd,opName)


                        /**
                         * TODO: Figure out if post processing is needed.
                         *
                         */
                        //DType calculate for output variables (set/correct if necessary)
                        val newInNames = sd.ops[name]!!.inputsToOp //Just in case import has modified this, like for concat case
                        val newInDtypes: MutableList<DataType> =
                            java.util.ArrayList(newInNames.size)
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
                            throw java.lang.IllegalStateException("Op $nd4jOpName does not have any outputs!")
                        }

                        //println("Out dtypes size ${outDTypes.size} and numOutputs $numOutputs")
                        val outSDVars = arrayOfNulls<SDVariable>(numOutputs)
                        val outVars = arrayOfNulls<Variable>(numOutputs)
                        val outNames: MutableList<String> = java.util.ArrayList(numOutputs)

                        //Create output variables and add to graph
                        for (i in 0 until numOutputs) {
                            val dt = outputDataTypes[i]
                            val varName = name + if (i == 0) "" else ":$i"
                            //TODO: handle variadic type in kotlin
                            /**
                             * TODO: handle data type import
                             */
                            outSDVars[i] = sd.`var`(varName, VariableType.ARRAY, null, dt)
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
                        }
                        sd.ops[name]!!.outputsOfOp = outNames
                        println("Imported op: $opName (name=$name)")
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
                    val mappingContext = irGraph.createMappingContext(
                        opDef = opDefLookup,
                        node = irGraph.nodeByName(name),
                        dynamicVariables = dynamicVariables
                    )

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

                        o!!.initAttributes(df,irGraph.frameworkName(),mappingContext,sd,opName)
                    }
                }
            }


            //Now that we have just added an op (or variable) - check what this feeds into, and see what we can now process
            // as a result
            if (nodeInputTo.containsKey(name)) {
                val set: List<String>? = nodeInputTo[name]
                for (nextOp in set!!) {
                    val nextOpDef = remainingNodes[nextOp]
                    if(nextOpDef == null)
                        throw java.lang.IllegalStateException("No next op def found for op $nextOp")
                    val nInNext = nextOpDef.numInputs()

                    if (nextOpDef == null) {
                        if (sd.ops.containsKey(nextOp)) {
                            //Already processed this.
                            //Almost certainly the close of a loop - like NextIteration -> Merge case
                            continue
                        }
                        throw IllegalStateException("Could not find op definition for op to import: $nextOp")
                    }
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
        }

        //Post process the control dependencies, if any (done after because dependencies may not exist when imported)
        for ((varName, cdOpNames) in constControlDeps) {
            sd.variables[varName]!!.controlDeps = cdOpNames
            for (s in cdOpNames) {
                val sdo = sd.ops[s]
                if (sdo!!.controlDepFor == null) sdo.controlDepFor = java.util.ArrayList()
                val l = sdo.controlDepFor
                if (!l.contains(s)) l.add(varName)
            }
        }

        //Post process the merge ops - all we are missing is a Variable.getInputsForOp().add(mergeOpName);
        for ((key, value) in mergeOpsPostProcess) {
            val v = sd.variables[value]
            if (v!!.inputsForOp == null) v.inputsForOp = java.util.ArrayList()
            v.inputsForOp.add(key)
        }
        Preconditions.checkState(
            remainingNodes.isEmpty(),
            "%s Unprocessed nodes: %s",
            remainingNodes.size,
            remainingNodes.keys
        )
        return sd
    }
}

