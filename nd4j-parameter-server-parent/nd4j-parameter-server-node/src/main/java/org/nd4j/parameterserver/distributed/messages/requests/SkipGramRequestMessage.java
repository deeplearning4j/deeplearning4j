package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.parameterserver.distributed.logic.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedDotMessage;
import org.nd4j.parameterserver.distributed.training.TrainerProvider;

/**
 * This is batch message, describing simple SkipGram round
 *
 * We assume this message is created on Client, and passed to selected Shard
 * Shard which received this message becomes a driver, which handles processing
 *
 * @author raver119@gmail.com
 */
@Data
public class SkipGramRequestMessage extends BaseVoidMessage implements TrainingMessage {

    // learning rate for this sequence
    protected float alpha;

    // current word & lastWord
    protected int w1;
    protected int w2;

    // following fields are for hierarchic softmax
    // points & codes for current word
    protected int[] points;
    protected byte[] codes;

    protected short negSamples;

    protected SkipGramRequestMessage() {
        super(0);
    }

    public SkipGramRequestMessage(int w1, int w2, int[] points, byte[] codes, short negSamples) {
        this();
        this.w1 = w1;
        this.w2 = w2;
        this.points = points;
        this.codes = codes;
        this.negSamples = negSamples;
    }

    /**
     * This method does actual training for SkipGram algorithm
     */
    @Override
    public void processMessage() {
        /**
         * This method in reality just delegates training to specific TrainingDriver, based on message type.
         * In this case - SkipGram training
         */
        TrainerProvider.getInstance().doTraining(this);
    }
}
