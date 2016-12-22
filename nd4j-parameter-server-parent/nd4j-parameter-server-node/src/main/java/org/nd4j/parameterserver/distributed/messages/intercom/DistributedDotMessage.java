package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;

/**
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class DistributedDotMessage extends BaseVoidMessage {

    protected long taskId;
    protected Integer keyA;
    protected Integer keyB;
    protected int[] rowsA;
    protected int[] rowsB;

    public DistributedDotMessage() {
        messageType = 22;
    }


    public DistributedDotMessage(long taskId, Integer keyA, Integer keyB, int rowA, int rowB) {
        this(taskId, keyA, keyB, new int[]{rowA}, new int[]{rowB});
    }

    public DistributedDotMessage(long taskId, @NonNull Integer keyA, @NonNull Integer keyB, @NonNull int[] rowsA, @NonNull int[] rowsB) {
        this();
        this.keyA = keyA;
        this.keyB = keyB;
        this.rowsA = rowsA;
        this.rowsB = rowsB;
        this.taskId = taskId;

        if (this.rowsA.length != this.rowsB.length)
            throw new ND4JIllegalStateException("Length of X should match length of Y");
    }

    /**
     * This method calculates dot of gives rows
     */
    @Override
    public void processMessage() {
        //TODO: make this thing a single op, even specialOp is ok
        // we calculate dot for all involved rows
        INDArray result = Nd4j.createUninitialized(rowsA.length, 1);
        for (int e = 0; e < rowsA.length; e++) {
            double dot = Nd4j.getBlasWrapper().dot(storage.getArray(keyA).getRow(rowsA[e]), storage.getArray(keyB).getRow(rowsB[e]));
            result.putScalar(e, dot);
        }

        // send this message to everyone
        DotAggregation dot = new DotAggregation(taskId, (short) configuration.getNumberOfShards(), shardIndex, result);
        transport.sendMessage(dot);
    }
}
