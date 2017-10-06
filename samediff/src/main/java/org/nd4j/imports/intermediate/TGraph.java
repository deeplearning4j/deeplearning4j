package org.nd4j.imports.intermediate;

import com.google.common.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.graph.FlatGraph;
import org.nd4j.graph.FlatNode;
import org.nd4j.graph.FlatVariable;
import org.nd4j.graph.IntPair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class provides intermediate representation of Graph
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class TGraph {
    @Getter protected TVariableSpace variableSpace = new TVariableSpace();

    // this is the layered representation
    protected Map<Integer, List<TNode>> onionMap = new HashMap<>();

    protected Map<Integer, TNode> outputMap = new HashMap<>();

    // here we're storing unmapped nodes
    protected List<TNode> unmapped = new ArrayList<>();

    protected void expandOnion(int layer) {
        onionMap.put(layer, new ArrayList<>());
    }

    public TNode getNode(@NonNull Integer index) {
        return outputMap.get(index);
    }

    public void addNode(@NonNull TNode node) {
        unmapped.add(node);
        outputMap.put(node.getId(), node);
    }

    protected int getTailSize() {
        return unmapped.size();
    }

    protected void buildOnion() {
        while (getTailSize() > 0) {

        }
    }

    public TGraph provideArrayForVariable(String id, INDArray array) {
        if (!variableSpace.hasVariable(id))
            throw new ND4JIllegalStateException("Unknown variable provided: [" + id + "]");

        variableSpace.getVariable(id).setArray(array);

        return this;
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

            val arr = variable.getArray();
            if (arr == null)
                continue;

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

        // and now we're dumping unmapped nodes, just in case of...
        for (val node: unmapped) {
            log.info("Exporting node: [{}]", node.getOpName());

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
            int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, node.getOpState().getOpType() == OpState.OpType.CUSTOM && node.getOpState().getExtraBits() != null ? node.getOpState().getExtraBits() : new int[]{});
            int dimensions = FlatNode.createDimensionsVector(bufferBuilder, node.getOpState().getAxes() != null ? node.getOpState().getAxes() : new int[]{});
            int fname = bufferBuilder.createString(node.getName());

            int flatNode = FlatNode.createFlatNode(bufferBuilder,
                    node.getId(),
                    fname,
                    NativeGraphExecutioner.getFlatOpType(node.getOpState().getOpType()),
                    NativeGraphExecutioner.getOpNum(node.getOpState().getOpName(), node.getOpState().getOpType()),
                    nodesIn,
                    nodesInPaired,
                    (byte) 0,
                    nodesOut,
                    extraz,
                    integerArgs,
                    dimensions,
                    -1,
                    node.getOpState().getOpType() == OpState.OpType.SCALAR_TRANSFORM ? node.getOpState().getScalarValue().floatValue() : 0.0f);

            flatNodes.add(flatNode);
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
}
