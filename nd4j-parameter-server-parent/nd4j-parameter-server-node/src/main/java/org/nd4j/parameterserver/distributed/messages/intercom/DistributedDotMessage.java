package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.training.impl.SkipGramTrainer;

/**
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class DistributedDotMessage extends BaseVoidMessage implements DistributedMessage {
    protected Integer keyA;
    protected Integer keyB;
    protected int[] rowsA;
    protected int[] rowsB;

    // payload for trainer pickup
    protected int w1, w2;
    protected boolean useHS;
    protected short negSamples;
    protected float alpha;
    protected byte[] codes;

    public DistributedDotMessage() {
        messageType = 22;
    }

    @Deprecated
    public DistributedDotMessage(long taskId, Integer keyA, Integer keyB, int rowA, int rowB) {
        this(taskId, keyA, keyB, new int[]{rowA}, new int[]{rowB}, 0, 0, new byte[]{}, false, (short) 0, 0.001f);
    }

    public DistributedDotMessage(long taskId,
                                 @NonNull Integer keyA,
                                 @NonNull Integer keyB,
                                 @NonNull int[] rowsA,
                                 @NonNull int[] rowsB,
                                 int w1,
                                 int w2,
                                 @NonNull byte[] codes,
                                 boolean useHS,
                                 short negSamples,
                                 float alpha
                                 ) {
        this();
        this.keyA = keyA;
        this.keyB = keyB;
        this.rowsA = rowsA;
        this.rowsB = rowsB;
        this.taskId = taskId;

        this.w1 = w1;
        this.w2 = w2;
        this.useHS = useHS;
        this.negSamples = negSamples;
        this.alpha = alpha;
        this.codes = codes;


        if (this.rowsA.length != this.rowsB.length)
            throw new ND4JIllegalStateException("Length of X should match length of Y");
    }

    /**
     * This method calculates dot of gives rows
     */
    @Override
    public void processMessage() {
        // this only picks up new training round
        SkipGramRequestMessage sgrm = new SkipGramRequestMessage(w1, w2, rowsB, codes, negSamples, alpha, 119 );

        // FIXME: get rid of THAT
        SkipGramTrainer sgt = (SkipGramTrainer) trainer;
        sgt.pickTraining(sgrm);

        //TODO: make this thing a single op, even specialOp is ok
        // we calculate dot for all involved rows
        INDArray result = Nd4j.createUninitialized(rowsA.length, 1);
        for (int e = 0; e < rowsA.length; e++) {
            double dot = Nd4j.getBlasWrapper().dot(storage.getArray(keyA).getRow(rowsA[e]), storage.getArray(keyB).getRow(rowsB[e]));
            result.putScalar(e, dot);
        }

        // send this message to everyone
        DotAggregation dot = new DotAggregation(taskId, (short) voidConfiguration.getNumberOfShards(), shardIndex, result);
        transport.sendMessage(dot);
    }
}
