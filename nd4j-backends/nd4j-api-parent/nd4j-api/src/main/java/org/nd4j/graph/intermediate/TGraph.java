package org.nd4j.graph.intermediate;

import com.google.common.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.graph.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This class provides intermediate representation of Graph
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class TGraph {
    @Getter protected TVariableSpace variableSpace = new TVariableSpace();

    // this map contains reverse lookup information, between external graph (i.e. tf) and ours
    @Getter protected Map<String, TIndex> reverseMap = new HashMap<>();

    // this is the layered representation
    protected Map<Integer, List<TOp>> onionMap = new HashMap<>();

    protected Map<Integer, TOp> outputMap = new HashMap<>();
    protected Map<String, TOp> symbolicMap = new HashMap<>();

    // here we're storing unmapped nodes
    protected List<TOp> unmapped = new ArrayList<>();

    // storage for Scopes
    protected Map<Integer, TScope> numericScopes = new HashMap<>();
    protected Map<String, TScope> symbolicScopes = new HashMap<>();

    // counters for import processs
    protected AtomicInteger varsCnt = new AtomicInteger(0);
    protected AtomicInteger nodesCnt = new AtomicInteger(0);

    // here we store nodes which were already processed by
    @Getter protected Collection<String> skipSet = new ArrayList<>();

    @Getter protected Collection<String> knownScopes = new ArrayList<>();

    protected void expandOnion(int layer) {
        onionMap.put(layer, new ArrayList<TOp>());
    }

    public TOp getNode(@NonNull Integer index) {
        return outputMap.get(index);
    }

    public TOp getNode(@NonNull String name) {
        return symbolicMap.get(name);
    }

    public void addNode(@NonNull TOp node) {
        unmapped.add(node);
        outputMap.put(node.getId(), node);

        if (node.getName() != null && !node.getName().isEmpty()) {
            log.info("Adding node by opName: [{}]", node.getName());
            symbolicMap.put(node.getName(), node);
        }
    }

    /**
     * This method returns current node id, without increment
     * @return
     */
    public int getCurrentNodeId() {
        return nodesCnt.get();
    }

    /**
     * This method returns new node id, pre-increment
     * @return
     */
    public int getNewNodeId() {
        return nodesCnt.incrementAndGet();
    }

    /**
     * This method returns current var id, without decrement
     * @return
     */
    public int getCurrentVariableId() {
        return varsCnt.get();
    }

    /**
     * This method returns new node id, pre-decrement
     * @return
     */
    public int getNewVariableId() {
        return varsCnt.decrementAndGet();
    }

    protected int getTailSize() {
        return unmapped.size();
    }

    protected void buildOnion() {
        while (getTailSize() > 0) {

        }
    }

    /**
     * This mehtod adds Scope to this graph
     * @param scope
     */
    public void addScope(@NonNull TScope scope) {
        numericScopes.put(scope.getId(), scope);
        symbolicScopes.put(scope.getName(), scope);
    }

    /**
     * This method returns Scope by symbolic opName
     *
     * @param name
     * @return
     */
    public TScope getScope(@NonNull String name) {
        if (!symbolicScopes.containsKey(name))
            throw new ND4JIllegalStateException("No scope with given opName found: [" + name + "]");

        return symbolicScopes.get(name);
    }

    /**
     * This method returns Scope by id
     *
     * @param id
     * @return
     */
    public TScope getScope(int id) {
        if (!numericScopes.containsKey(id))
            throw new ND4JIllegalStateException("No scope with given opName found: [" + id + "]");

        return numericScopes.get(id);
    }

    public TGraph provideArrayForVariable(String id, INDArray array) {
        if (!variableSpace.hasVariable(id))
            throw new ND4JIllegalStateException("Unknown variable provided: [" + id + "]");

        variableSpace.getVariable(id).setArray(array);

        return this;
    }

    protected int asFlatNode(@NonNull TScope scope, @NonNull FlatBufferBuilder bufferBuilder) {

        int scopeName = bufferBuilder.createString(scope.getName());

        int flatNode = FlatNode.createFlatNode(bufferBuilder,
                scope.getId(),
                scopeName,
                OpType.LOGIC,
                10, // hardcoded value
                0,
                0,
                (byte) 0,
                0,
                0,
                0,
                0,
                -1,
                0.0f, 0, 0);

        return flatNode;
    }

    protected int asFlatNode(@NonNull TOp node, @NonNull FlatBufferBuilder bufferBuilder) {
        log.info("Exporting node: [{}:<{}>]", node.getOpName(), node.getName());

        float[] extras = node.getOpState().getExtraArgs() != null ? new float[node.getOpState().getExtraArgs().length] : new float[0];
        for (int e = 0; e < extras.length; e++) {
            extras[e] = ((Number) node.getOpState().getExtraArgs()[e]).floatValue();
        }

        val inPaired = new ArrayList<Integer>();
        int e = 0;
        for (val index: node.getInputs())
            inPaired.add(IntPair.createIntPair(bufferBuilder, index.getNode(), index.getIndex()));

        int nodesIn = FlatNode.createInputVector(bufferBuilder, new int[]{});
        int nodesInPaired = FlatNode.createInputPairedVector(bufferBuilder, Ints.toArray(inPaired));
        int nodesOut = FlatNode.createOutputVector(bufferBuilder, Ints.toArray(node.getOutputs()));
        int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
        int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, node.getOpState().getOpType() == Op.Type.CUSTOM && node.getOpState().getExtraBits() != null ? node.getOpState().getExtraBits() : new int[]{});
        int dimensions = FlatNode.createDimensionsVector(bufferBuilder, node.getOpState().getAxes() != null ? node.getOpState().getAxes() : new int[]{});
        int fname = bufferBuilder.createString(node.getName());
        int scopeName = bufferBuilder.createString(node.getScopeName());

        if (node.getOpState().getOpType() == null)
            log.warn("Null-op node: {}", node);

        int flatNode = FlatNode.createFlatNode(bufferBuilder,
                node.getId(),
                fname,
                getFlatOpType(node.getOpState().getOpType()),
                getOpNum(node.getOpState().getOpName(), node.getOpState().getOpType()),
                nodesIn,
                nodesInPaired,
                (byte) 0,
                nodesOut,
                extraz,
                integerArgs,
                dimensions,
                -1,
                node.getOpState().getOpType() == Op.Type.SCALAR ? node.getOpState().getScalarValue().floatValue() : 0.0f, node.getScopeId(), scopeName);

        return flatNode;
    }

    public ByteBuffer asFlatBuffers() {
        if (variableSpace.hasUndefinedPlaceholders())
            throw new ND4JIllegalStateException("You should provide placeholder values before launching graph");

        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(1024);

        val flatVariables = new ArrayList<Integer>();
        val flatOffsets = new ArrayList<Integer>();
        val flatNodes = new ArrayList<Integer>();

        // first of all we build VariableSpace dump
        for (val variable: variableSpace.getAllVariables()) {
            log.info("Exporting variable: [{}]", variable.getName());


            if (variable.getArray() == null) {
                if (variable.getShape() == null)
                    throw new ND4JIllegalStateException("Both array and shape are NULL");

                variable.setArray(Nd4j.create(variable.getShape()));
            }
            val arr = variable.getArray();


            int name = bufferBuilder.createString(variable.getName());
            int values = FlatVariable.createValuesVector(bufferBuilder, arr.data().asFloat());
            int shape = FlatVariable.createShapeVector(bufferBuilder, arr.shapeInfoDataBuffer().asInt());

            int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, variable.getId(), name, shape, values, -1);
            flatVariables.add(flatVariable);

        }


        // then we build onion dump. we don't need it, but why not?
        val keys = onionMap.keySet();
        for (val key: keys) {
            val ops = onionMap.get(key);

            for (val node: ops) {
                // dump right here
            }
        }

        // we're dumping scopes now
        for (val scope: numericScopes.values()) {
            flatNodes.add(asFlatNode(scope, bufferBuilder));

            // converting all ops from node
            for (val node: scope.getNodes()) {
                flatNodes.add(asFlatNode(node, bufferBuilder));
            }
        }



        // and now we're dumping unmapped nodes, just in case of...
        for (val node: unmapped) {
            flatNodes.add(asFlatNode(node, bufferBuilder));
        }

        int outputsOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatOffsets));
        int variablesOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatVariables));
        int nodesOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(flatNodes));

        val configuration = ExecutorConfiguration.builder()
                .outputMode(OutputMode.IMPLICIT)
                .executionMode(ExecutionMode.SEQUENTIAL)
                .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
                .gatherTimings(true)
                .build();

        int fg = FlatGraph.createFlatGraph(bufferBuilder, 119, variablesOffset, nodesOffset, outputsOffset, configuration.getFlatConfiguration(bufferBuilder));
        bufferBuilder.finish(fg);

        return bufferBuilder.dataBuffer();
    }

    public static long getOpNum(String name, Op.Type type) {
        if (type == Op.Type.LOOP ) {
            return 0;
        } else if (type == Op.Type.RETURN) {
            return 40;
        } else if (type == Op.Type.IF || type == Op.Type.CONDITIONAL) {
            return 10;
        } else if (type == Op.Type.CUSTOM)
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
            case SPECIAL:
                return OpType.TRANSFORM;
            case REDUCE:
                return OpType.ACCUMULATION;
            case INDEXREDUCE:
                return OpType.INDEX_ACCUMULATION;
            case LOOP:
                return OpType.LOGIC;
            case RETURN:
                return OpType.LOGIC;
            case CUSTOM:
                return OpType.CUSTOM;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }
}
