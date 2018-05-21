package org.nd4j.autodiff.execution;

import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
public class BasicGraphExecutioner implements GraphExecutioner {
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
     * @param graph
     * @return
     */
    @Override
    public INDArray[] executeGraph(SameDiff graph, ExecutorConfiguration configuration) {
        return new INDArray[]{graph.execAndEndResult()};
    }

    /**
     *
     * @param diff
     * @return
     */
    public ByteBuffer convertToFlatBuffers(SameDiff diff, ExecutorConfiguration configuration) {
        throw new UnsupportedOperationException();
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
        return executeGraph(sd, new ExecutorConfiguration());
    }

    @Override
    public INDArray[] reuseGraph(SameDiff graph, Map<Integer, INDArray> inputs) {
        throw new UnsupportedOperationException();
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
        // TODO: to be implemented
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /**
     * This method stores given graph for future execution
     *
     * @param graph
     * @return
     */
    @Override
    public int registerGraph(SameDiff graph) {
        // TODO: to be implemented
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public INDArray[] importProto(File file) {
        // TODO: to be implemented
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
