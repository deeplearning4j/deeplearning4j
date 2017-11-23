package org.nd4j.autodiff.execution;

import com.google.common.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.HalfIndexer;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.*;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class NativeGraphExecutioner implements GraphExecutioner {
    /**
     * This method returns Type of this executioner
     *
     * @return
     */
    @Override
    public Type getExecutionerType() {
        return Type.LOCAL;
    }


    /**
     * This method executes given graph and returns results
     *
     * PLEASE NOTE: Default configuration is used
     *
     * @param sd
     * @return
     */
    @Override
    public INDArray[] executeGraph(SameDiff sd) {
        return executeGraph(sd, ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).executionMode(ExecutionMode.SEQUENTIAL).profilingMode(OpExecutioner.ProfilingMode.DISABLED).build());
    }

    public ByteBuffer convertToFlatBuffers(SameDiff sd, ExecutorConfiguration configuration, Map<Integer, Node> intermediate) {
        log.info("Configuration: {}", configuration);

        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(32);

        SDGraph graph =  sd.getGraph();
        List<Integer> variables = new ArrayList<>();

        log.info("Graph inputs: {}", graph.getInputs());

        int varId = 0;
        Map<String, int[]> mappedVariables = new HashMap<>();
        Map<int[], int[]> mappedInputs = new HashMap<>();

        // mapping input variables first
        for (SDVariable input: graph.getInputs()) {
            varId--;
            SDVariable sdVar = input;
            //log.info("Var: {}", sdVar);

            log.info("Input varId: {}; varName: {};", varId, input.getVarName());

            INDArray arr = sdVar.getArr().isView() ? sdVar.getArr().dup(sdVar.getArr().ordering()) : sdVar.getArr();
            int name = bufferBuilder.createString(sdVar.getVarName());
            int shape = FlatArray.createShapeVector(bufferBuilder, arr.shapeInfoDataBuffer().asInt());
            int buffer = FlatArray.createBufferVector(bufferBuilder, arr.data().asBytes());
            int id = IntPair.createIntPair(bufferBuilder, sdVar.getVertexId()[0], 0);

            int array = FlatArray.createFlatArray(bufferBuilder, shape, buffer, SameDiff.getDataTypeAsByte(arr.data().dataType()), SameDiff.getOrderAsByte());



            int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, 0, array, -1);
            variables.add(flatVariable);

            mappedVariables.put(input.getVarName(), new int[]{varId});
        }


        List<OpExecAction> ops = graph.getOpOrder().getActions();
        int nodesCount = 1;
        for (OpExecAction action: ops) {
            log.info("Action: {}", action);
            int[] ins = action.getInputsIds();

            Node node = new Node();
            node.setId(nodesCount);
            node.setName(Arrays.toString(action.getOutputId()));
            node.setOpExecAction(action);
            //node.setOriginalOutput(action.getOutputId());


            mappedInputs.put(action.getOutputId(), new int[]{nodesCount});

            // each of inputs can be either external variable, or another node
            for (int in: ins) {
                SDVariable state = sd.getVariableForVertexId(action.getOutputId());

                int[] realIn;
                if (state != null && mappedVariables.containsKey(state.getVarName())) {
                    // this means it's external variable, already available at mappedInputs
                    log.info("External input: {}", mappedVariables.get(state.getVarName()));
                    realIn = mappedVariables.get(state.getVarName());
                } else if (mappedInputs.containsKey(in)) {
                    log.info("Node as input: {}", mappedInputs.get(in));
                    realIn = mappedInputs.get(in);
                } else {
                    throw new RuntimeException("Unknown graph input node found: " + in);
                }

                // updating intermediate representation with mapped input
                node.getInput().addAll(Ints.asList(realIn));
            }

            intermediate.put(nodesCount, node);

            nodesCount++;
        }

        // now we'll update output fields with proper values
        Set<Integer> keySet = intermediate.keySet();
        for (Integer n: keySet) {
            Node node = intermediate.get(n);
            for (Integer in: node.getInput()) {

                // we're skipping variables here, since we already mapped them
                if (in >= 0) {
                    intermediate.get(in).getOutput().add(node.getId());
                }
            }
        }

        // building FlatBuffers now
        List<Integer> nodes = new ArrayList<>();
        for (Integer n: keySet) {
            Node node = intermediate.get(n);
            val func = sd.getFunctionForVertexId(node.getOriginalOutput());
            int[] extraBits = null;
            if(func.opType() == Op.Type.CUSTOM) {
               DynamicCustomOp dynamicCustomOp = (DynamicCustomOp) func;
               extraBits = Ints.toArray(dynamicCustomOp.getIArguments());
            }
            else
                extraBits = new int[]{};
            val extraArgs = sd.getFunctionForVertexId(node.getOriginalOutput()).getExtraArgs();
            // make this variable
            float[] extras = extraArgs != null ? new float[extraArgs.length] : new float[0];
            val opType = sd.getFunctionForVertexId(node.getOriginalOutput()).opType();
            val opName = sd.getFunctionForVertexId(node.getOriginalOutput()).opName();
            for (int e = 0; e < extras.length; e++) {
                extras[e] = ((Number) extraArgs[e]).floatValue();
            }

            int nodesIn = FlatNode.createInputVector(bufferBuilder, Ints.toArray(node.getInput()));
            int nodesInP = FlatNode.createInputPairedVector(bufferBuilder, new int[]{});
            int nodesOut = FlatNode.createOutputVector(bufferBuilder, Ints.toArray(node.getOutput()));
            int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
            int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, extraBits);
            int dimensions = FlatNode.createDimensionsVector(bufferBuilder,func.getDimensions() != null ? func.getDimensions() : new int[]{});
            int fname = bufferBuilder.createString(node.getName());

            int flatNode = FlatNode.createFlatNode(bufferBuilder,
                    node.getId(),
                    fname,
                    getFlatOpType(opType),
                    getOpNum(opName, opType),
                    nodesIn,
                    nodesInP,
                    (byte) 0,
                    nodesOut,
                    extraz,
                    integerArgs,
                    dimensions,
                    -1,
                    opType == Op.Type.SCALAR ? func.getScalarValue().floatValue() : 0.0f, 0, 0);

            nodes.add(flatNode);
        }

        log.info("-------");
        log.info("Intermediate: {}", intermediate);


        int outputsOffset = FlatGraph.createVariablesVector(bufferBuilder, new int[]{});

        int variablesOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(variables));
        int nodesOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(nodes));

        int fg = FlatGraph.createFlatGraph(bufferBuilder, 119, variablesOffset, nodesOffset, outputsOffset, configuration.getFlatConfiguration(bufferBuilder));
        bufferBuilder.finish(fg);

        return bufferBuilder.dataBuffer();
    }

    @Override
    public ByteBuffer convertToFlatBuffers(SameDiff sd, ExecutorConfiguration configuration) {
        return convertToFlatBuffers(sd, configuration, new HashMap<Integer, Node>());
    }

    /**
     * This method executes given graph and returns results
     *
     * @param sd
     * @return
     */
    @Override
    public INDArray[] executeGraph(SameDiff sd, ExecutorConfiguration configuration) {

        Map<Integer, Node> intermediate = new HashMap<>();

        ByteBuffer buffer = convertToFlatBuffers(sd, configuration, intermediate);

        BytePointer bPtr = new BytePointer(buffer);

        log.info("Buffer length: {}", buffer.limit());

        Pointer res  = NativeOpsHolder.getInstance().getDeviceNativeOps().executeFlatGraphFloat(null, bPtr);

        // FIXME: this is BAD
        PagedPointer pagedPointer = new PagedPointer(res,1024 * 1024L);
        FlatResult fr = FlatResult.getRootAsFlatResult(pagedPointer.asBytePointer().asByteBuffer());


        log.info("VarMap: {}", sd.variableMap());

        INDArray[] results = new INDArray[fr.variablesLength()];

        for (int e = 0; e < fr.variablesLength(); e++) {
            FlatVariable var = fr.variables(e);
            log.info("Var received: id: {}; opName: {}", var.id(), var.name());
            FlatArray ndarray = var.ndarray();
            val dtype = ndarray.dtype();
            int[] shape = new int[var.shapeLength()];

            for (int i = 0; i < ndarray.shapeLength(); i++) {
                shape[i] = ndarray.shape(i);
            }

            int[] _shape = new int[shape[0]];
            for (int i = 0; i < _shape.length; i++) {
                _shape[i] = shape[i+1];
            }

            val bufLen = ArrayUtil.prod(_shape);
            double[] values = new double[bufLen];

            val bb = ndarray.bufferAsByteBuffer();

            if (dtype == DataType.FLOAT) {
                val fb = bb.asFloatBuffer();
                for (int i = 0; i < bufLen; i++)
                    values[i] = fb.get(i);

            } else if (dtype == DataType.HALF) {
                val hb = bb.asShortBuffer();
                for (int i = 0; i < bufLen; i++)
                    values[i] = (double) HalfIndexer.toFloat((int) hb.get(i));

            } else if (dtype == DataType.DOUBLE) {
                val db = bb.asDoubleBuffer();
                for (int i = 0; i < bufLen; i++)
                    values[i] = db.get(i);
            }

            char _order = shape[shape[0] * 2 + 4 - 1] == 99 ? 'c' : 'f';

            INDArray val = Nd4j.create(values, _shape, _order, 0);
            results[e] = val;

            if (var.name() != null && sd.variableMap().containsKey(var.name())) {
                //log.info("VarName: {}; Exists: {}; NDArrayInfo: {};", var.opName(), sd.variableMap().containsKey(var.opName()), sd.getVertexToArray().containsKey(var.opName()));
                sd.associateArrayWithVariable(val, sd.variableMap().get(var.name()));

            } else {
                int[] original = intermediate.get(var.id()).getOriginalOutput();
                //log.info("Original id: {}; out: {}; out2: {}", original, sd.getVertexIdxToInfo().get(original), graph.getVariableForVertex(original));
                if (sd.variableMap().get(sd.getGraph().getVariableForVertex(original[0]).getVarName()) != null) {
                    sd.associateArrayWithVariable(val,sd.variableMap().get(sd.getGraph().getVariableForVertex(original[0]).getVarName()));
                } else {
                    SDVariable variable = SDVariable.builder()
                            .varName(sd.getGraph().getVariableForVertex(original[0]).getVarName())
                            .shape(val.shape())
                            .sameDiff(sd)
                            .build();

                    sd.associateArrayWithVariable(val,variable);
                    sd.addVariable(variable);
                }
            }
        }


        return results;
    }


    /*
    // first version

    @Override
    public INDArray[] executeGraph(SameDiff sd, ExecutorConfiguration configuration) {
        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(2048);

        SDGraph graph =  sd.getGraph();

        log.info("SameDiff variables: {}", sd.getSameDiffVariables());
        log.info("Graph outputs: {}", graph.getOutputs());
        log.info("--------------");

        // we use this map to convert SDVariables to op nodes for native backend
        Map<Integer, Integer> vertexMap = new HashMap<>();
        Map<String, Integer> vertexMapS = new HashMap<>();
        Map<Integer, List<Integer>> useMap = new HashMap<>();

        List<OpExecAction> ops = graph.getOpOrder().getActions();
        List<Integer> nodes = new ArrayList<>();
        List<Integer> variables = new ArrayList<>();
        int nodesCount = 1;

        // in first loop we build vertexMap only for output nodes
        for (OpExecAction action: ops) {
            log.info("Action: {}", action);
            NDArrayInformation out = action.getOutput();
            SDVariable sdOut = sd.variableMap().get(out.getId());

            // output of this operation is declared variable
            if (sdOut != null && sdOut.getId() < 0) {
                vertexMapS.put(out.getId(), sdOut.getId());
                log.info("Storing [{}/{}] variable as node_{} output", action.getOutputId(), out.getId(), nodesCount);
            } else {
                // output of this node is internal variable, we'll assume this node everywhere
                vertexMap.put(action.getOutputId(), nodesCount);
                vertexMapS.put(out.getId(), nodesCount);
                log.info("Storing [{}/{}] variable as node_{} output", action.getOutputId(), out.getId(), nodesCount);
            }

            if (useMap.get(nodesCount) == null)
                useMap.put(nodesCount, new ArrayList<>());

            nodesCount++;
        }

        log.info("-------------------");

        // in this loop we build list of input nodes
        nodesCount = 1;
        for (OpExecAction action: ops) {

            for (NDArrayInformation var: action.getInputs()) {
                SDVariable sdVar = sd.variableMap().get(var.getId());

                log.info("Var: {}; Mapping {} to node: {}", var.getId(), vertexMapS.get(var.getId()), nodesCount);

                if (sdVar != null && sdVar.getId() >= 0)
                    useMap.get(vertexMapS.get(var.getId())).add(nodesCount);
            }

            nodesCount++;
        }

        log.info("-------------------");

        // in this loop we build nodes
        nodesCount = 1;
        for (OpExecAction action: ops) {
            log.info("Op: {}", action.getOpState());

            int[] mappedIns = new int[action.getInputs().length];

            // meh
            int[] mappedOuts = new int[useMap.get(nodesCount).size()];


            int varsCount = 0;
            // fetching input vars first
            for (NDArrayInformation var: action.getInputs()) {
                SDVariable sdVar = sd.variableMap().get(var.getId());

                // negative ID assumes pre-created array
                if (sdVar !=  null && sdVar.getId() < 0) {
                    log.info("Input varId: {}; varName: {};", sdVar.getId(), var.getId());

                    INDArray arr = sdVar.getArr().isView() ? sdVar.getArr().dup(sdVar.getArr().ordering()) : sdVar.getArr();
                    int opName = bufferBuilder.createString(sdVar.getVarName());
                    int values = FlatVariable.createValuesVector(bufferBuilder, arr.data().asFloat());
                    int shape = FlatVariable.createShapeVector(bufferBuilder, arr.shapeInfoDataBuffer().asInt());

                    int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, sdVar.getId(), opName, shape, values, -1);
                    variables.add(flatVariable);

                    mappedIns[varsCount++] = sdVar.getId();
                } else {
                    log.info("Empty Input varId: {}; varName: {};", vertexMapS.get(var.getId()), var.getId());

                    // in all other cases - it's "virtual" array, will be created as op result instead
                    int opName = bufferBuilder.createString("auto");
                    int values = FlatVariable.createValuesVector(bufferBuilder, new float[]{});
                    int shape = FlatVariable.createShapeVector(bufferBuilder, new int[]{});

                    // FIXME: we need auto ID here instead of 119
                    int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, 119, opName, shape, values, -1);
                    variables.add(flatVariable);

                    mappedIns[varsCount++] = vertexMapS.get(var.getId());
                }
            }

            int outCount = 0;
            for (Integer o : useMap.get(nodesCount)) {
                mappedOuts[outCount++] = o;
            }

            // make this variable
            float[] extras = action.getOpState().getExtraArgs() != null ? new float[action.getOpState().getExtraArgs().length] : new float[0];
            for (int e = 0; e < extras.length; e++) {
                extras[e] = ((Number) action.getOpState().getExtraArgs()[e]).floatValue();
            }

            log.info("Node_{} inputs: {}; outputs: {}", nodesCount, Arrays.toString(mappedIns), Arrays.toString(mappedOuts));
            int nodesIn = FlatNode.createInputVector(bufferBuilder, mappedIns);
            int nodesOut = FlatNode.createOutputVector(bufferBuilder, mappedOuts);
            int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
            int dimensions = FlatNode.createDimensionsVector(bufferBuilder, action.getOpState().getAxes() != null ? action.getOpState().getAxes() : new int[]{});

            int flatNode = FlatNode.createFlatNode(bufferBuilder,
                                                   nodesCount,
                                                   getFlatOpType(action.getOpState().getOpType()),
                                                   getOpNum(action.getOpState().getOpName(), action.getOpState().getOpType()),
                                                   nodesIn,
                                                   (byte) 0,
                                                   nodesOut,
                                                   extraz,
                                                   dimensions,
                                            -1,
                    action.getOpState().getOpType() == OpState.OpType.SCALAR_TRANSFORM ? action.getOpState().getScalarValue().floatValue() : 0.0f);

            nodes.add(flatNode);
            nodesCount++;
        }

        log.info("Variables: {}", variables);
        log.info("Nodes: {}", nodes);

        int outputsOffset = FlatGraph.createVariablesVector(bufferBuilder, new int[]{});
        int variablesOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(variables));
        int nodesOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(nodes));

        int fg = FlatGraph.createFlatGraph(bufferBuilder, 119, variablesOffset, nodesOffset, outputsOffset, configuration.getFlatConfiguration(bufferBuilder));
        bufferBuilder.finish(fg);

        ByteBuffer buffer = bufferBuilder.dataBuffer();
        BytePointer bPtr = new BytePointer(buffer);

        log.info("Buffer length: {}", buffer.limit());

        Pointer res  = NativeOpsHolder.getInstance().getDeviceNativeOps().executeFlatGraphFloat(null, bPtr);

        PagedPointer pagedPointer = new PagedPointer(res,      1024 * 1024L);
        FlatResult fr = FlatResult.getRootAsFlatResult(pagedPointer.asBytePointer().asByteBuffer());

        INDArray[] results = new INDArray[fr.variablesLength()];
        for (int e = 0; e < fr.variablesLength(); e++) {
            FlatVariable var = fr.variables(e);
            float[] values = new float[var.valuesLength()];
            int[] shape = new int[var.shapeLength()];

            for (int i = 0; i < var.valuesLength(); i++) {
                values[i] = var.values(i);
            }

            for (int i = 0; i < var.shapeLength(); i++) {
                shape[i] = var.shape(i);
            }

            int[] _shape = new int[shape[0]];
            for (int i = 0; i < _shape.length; i++) {
                _shape[i] = shape[i+1];
            }

            char _order = shape[shape[0] * 2 + 4 - 1] == 99 ? 'c' : 'f';

            INDArray val = Nd4j.create(values, _shape, _order, 0);
            results[e] = val;
        }

        return results;
    }
    */

    public static long getOpNum(String name, Op.Type type) {
        if (type == Op.Type.CUSTOM)
            return Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase()).getHash();
        else
            return (long) Nd4j.getOpFactory().getOpNumByName(name);
    }

    public static byte getFlatOpType(Op.Type type) {
        switch (type) {
            case SCALAR:
                return OpType.SCALAR;
            case BROADCAST:
                return OpType.BROADCAST;
            case TRANSFORM:
                return OpType.TRANSFORM;
            case REDUCE:
                return OpType.ACCUMULATION;
            case INDEXREDUCE:
                return OpType.INDEX_ACCUMULATION;
            case CUSTOM:
                return OpType.CUSTOM;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }

    /**
     * This method executes
     *
     * @param id
     * @param variables
     * @return
     */
    @Override
    public INDArray[] executeGraph(int id, SDVariable... variables) {
        return new INDArray[0];
    }

    /**
     * This method stores given graph for future execution
     *
     * @param graph
     * @return
     */
    @Override
    public int registerGraph(SameDiff graph) {
        return 0;
    }


    @Override
    public INDArray[] importProto(File file) {
        // TODO: to be implemented
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
