package org.nd4j.autodiff.execution;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.graph.FlatGraph;
import org.nd4j.graph.FlatNode;
import org.nd4j.graph.FlatVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

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


    protected Triple<FlatNode, FlatVariable[], FlatVariable[]> getFlatNodeFromOpState(OpState state) {

        return null;
    }

    /**
     * This method executes given graph and returns results
     *
     * @param sd
     * @return
     */
    @Override
    public INDArray[] executeGraph(SameDiff sd) {
        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(1024);

        SDGraph graph =  sd.getGraph();

        log.info("{}", sd.getSameDiffVariables());

        List<OpExecAction> ops = graph.getOpOrder().getActions();
        List<FlatNode> nodes = new ArrayList<>();
        int cnt = 1;
        for (OpExecAction action: ops) {
            Triple<FlatNode, FlatVariable[], FlatVariable[]> triple = getFlatNodeFromOpState(action.getOpState());


            nodes.add(triple.getFirst());
            cnt++;
        }

        FlatGraph fg = null; //FlatGraph.createFlatGraph(bufferBuilder);


        ByteBuffer buffer = fg.getByteBuffer();
        Pointer ptr = new Pointer(buffer);

        return new INDArray[0];
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
}
