package org.nd4j.parameterserver.distributed.training.impl;

import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.messages.requests.CbowRequestMessage;
import org.nd4j.parameterserver.distributed.training.BaseTrainer;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;

/**
 * @author raver119@gmail.com
 */
public class CbowTrainer extends BaseTrainer<CbowRequestMessage> {
    @Override
    public void startTraining(CbowRequestMessage message) {

    }

    @Override
    public void pickTraining(CbowRequestMessage message) {

    }

    @Override
    public void aggregationFinished(VoidAggregation aggregation) {

    }

    @Override
    public void finishTraining(long originatorId, long taskId) {

    }

    @Override
    public void addCompletionHook(long originatorId, long frameId, long messageId) {

    }

    @Override
    public String targetMessageClass() {
        return CbowRequestMessage.class.getSimpleName();
    }
}
