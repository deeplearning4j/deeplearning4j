package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.logic.sequence.BasicSequenceProvider;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;

import java.util.Arrays;

/**
 * This is batch message, describing simple SkipGram round
 *
 * We assume this message is created on Client, and passed to selected Shard
 * Shard which received this message becomes a driver, which handles processing
 *
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class SkipGramRequestMessage extends BaseVoidMessage implements TrainingMessage, RequestMessage {

    // learning rate for this sequence
    protected double alpha;

    long frameId;

    // current word & lastWord
    protected int w1;
    protected int w2;

    // following fields are for hierarchic softmax
    // points & codes for current word
    protected int[] points;
    protected byte[] codes;

    protected int[] negatives;

    protected short negSamples;

    protected long nextRandom;

    protected byte counter = 1;

    protected SkipGramRequestMessage() {
        super(0);
    }

    public SkipGramRequestMessage(int w1, int w2, int[] points, byte[] codes, short negSamples, double lr,
                    long nextRandom) {
        this();
        this.w1 = w1;
        this.w2 = w2;
        this.points = points;
        this.codes = codes;
        this.negSamples = negSamples;
        this.alpha = lr;
        this.nextRandom = nextRandom;

        // FIXME: THIS IS TEMPORARY SOLUTION - FIX THIS!!!1
        this.setTaskId(BasicSequenceProvider.getInstance().getNextValue());
    }

    /**
     * This method does actual training for SkipGram algorithm
     */
    @Override
    @SuppressWarnings("unchecked")
    public void processMessage() {
        /**
         * This method in reality just delegates training to specific TrainingDriver, based on message opType.
         * In this case - SkipGram training
         */
        //log.info("sI_{} starts SkipGram round...", transport.getShardIndex());

        // FIXME: we might use something better then unchecked opType cast here
        TrainingDriver<SkipGramRequestMessage> sgt = (TrainingDriver<SkipGramRequestMessage>) trainer;
        sgt.startTraining(this);
    }

    @Override
    public boolean isJoinSupported() {
        return true;
    }

    @Override
    public void joinMessage(VoidMessage message) {
        // TODO: apply proper join handling here
        counter++;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        if (!super.equals(o))
            return false;

        SkipGramRequestMessage message = (SkipGramRequestMessage) o;

        if (w1 != message.w1)
            return false;
        if (w2 != message.w2)
            return false;
        if (negSamples != message.negSamples)
            return false;
        if (!Arrays.equals(points, message.points))
            return false;
        return Arrays.equals(codes, message.codes);
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + w1;
        result = 31 * result + w2;
        result = 31 * result + Arrays.hashCode(points);
        result = 31 * result + Arrays.hashCode(codes);
        result = 31 * result + (int) negSamples;
        return result;
    }
}
