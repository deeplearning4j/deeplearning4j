package org.nd4j.parameterserver.distributed.training.impl;

import org.nd4j.parameterserver.distributed.logic.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedDotMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.training.BaseTrainer;

/**
 * Distributed SkipGram trainer
 *
 * @author raver119@gmail.com
 */
public class SkipGramTrainer extends BaseTrainer<SkipGramRequestMessage> {
    @Override
    public void doTraining(SkipGramRequestMessage message) {
        /**
         * 1) Calculate dot
         * 2) Calculate gradient
         * 3) Apply updates
         */

        /**
         * If we're on HS, we know pairs in advance, if not - we should use rng to get them
         */
        if (message.getPoints().length > 0) {
            // we assume this is HS round
            int row_syn0[] = replicate(message.getW1(), message.getPoints().length);

            // as result, we'll have aggregated dot as single ordered column, which might be used for gradient calculation
            DistributedDotMessage ddm = new DistributedDotMessage(0L, WordVectorStorage.SYN_0, WordVectorStorage.SYN_1, row_syn0, message.getPoints());
            transport.sendMessage(ddm);
        } else {
            // pure negSampling round

        }
    }

    @Override
    public String targetMessageClass() {
        return SkipGramRequestMessage.class.getSimpleName();
    }
}
