package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.parameterserver.distributed.logic.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedDotMessage;

/**
 * This is batch message, describing simple SkipGram round
 *
 * We assume this message is created on Client, and passed to selected Shard
 * Shard which received this message becomes a driver, which handles processing
 *
 * @author raver119@gmail.com
 */
@Data
public class SkipGramRequestMessage extends BaseVoidMessage {

    // current word & lastWord
    protected int w1;
    protected int w2;

    // following fields are for hierarchic softmax
    // points & codes for current word
    protected int[] points;
    protected byte[] codes;

    protected SkipGramRequestMessage() {
        super(0);
    }

    public SkipGramRequestMessage(int w1, int w2, int[] points, byte[] codes) {
        this();
        this.w1 = w1;
        this.w2 = w2;
        this.points = points;
        this.codes = codes;
    }

    /**
     * This method does actual training for SkipGram algorithm
     */
    @Override
    public void processMessage() {
        /**
         * We go for computational phases
         * phase A) calculate gradients for different rows
         * phase B) apply gradients
         */

        /**
         * If we're on HS, we know pairs in advance, if not - we should use rng to get them
         */
        if (points.length > 0) {
            // we assume this is HS round
            int row_syn0[] = replicate(w1, points.length);

            // as result, we'll have aggregated dot as single ordered column, which might be used for gradient calculation
            DistributedDotMessage ddm = new DistributedDotMessage(0L, WordVectorStorage.SYN_0, WordVectorStorage.SYN_1, row_syn0, points);
            transport.sendMessage(ddm);
        } else {
            // pure negSampling round

        }
    }
}
