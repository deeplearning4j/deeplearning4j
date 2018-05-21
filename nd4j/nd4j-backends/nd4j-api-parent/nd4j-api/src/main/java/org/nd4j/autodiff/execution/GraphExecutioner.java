package org.nd4j.autodiff.execution;

import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.Map;

/**
 * This interface
 * @author raver119@gmail.com
 */
public interface GraphExecutioner {
    enum Type {
        /**
         * Executor runs on the same box
         */
        LOCAL,

        /**
         * Executor runs somewhere else
         */
        REMOTE,
    }

    /**
     * This method returns Type of this executioner
     *
     * @return
     */
    Type getExecutionerType();

    /**
     * This method executes given graph and returns results
     *
     * @param graph
     * @return
     */
    INDArray[] executeGraph(SameDiff graph, ExecutorConfiguration configuration);

    INDArray[] executeGraph(SameDiff graph);

    INDArray[] reuseGraph(SameDiff graph, Map<Integer, INDArray> inputs);

    /**
     * This method converts given SameDiff instance to FlatBuffers representation
     *
     * @param diff
     * @return
     */
    ByteBuffer convertToFlatBuffers(SameDiff diff, ExecutorConfiguration configuration);

    /**
     * This method executes
     * @param id
     * @param variables
     * @return
     */
    INDArray[] executeGraph(int id, SDVariable... variables);


    /**
     * This method stores given graph for future execution
     *
     * @param graph
     * @return
     */
    int registerGraph(SameDiff graph);

    /**
     * This method executes TF graph
     *
     * PLEASE NOTE: This feature is under development yet
     *
     * @param file
     * @return
     */
    INDArray[] importProto(File file);


}
