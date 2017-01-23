package org.nd4j.parameterserver.distributed.training.impl;

import org.nd4j.parameterserver.distributed.logic.completion.FrameCompletionHandler;
import org.nd4j.parameterserver.distributed.logic.completion.RequestDescriptor;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.requests.CbowRequestMessage;
import org.nd4j.parameterserver.distributed.training.BaseTrainer;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.training.chains.CbowChain;
import org.nd4j.parameterserver.distributed.training.chains.SkipGramChain;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class CbowTrainer extends BaseTrainer<CbowRequestMessage> {
    private static final float HS_MAX_EXP = 6.0f;

    protected Map<RequestDescriptor, CbowChain> chains = new ConcurrentHashMap<>();
    protected AtomicLong cntRounds = new AtomicLong(0);



    @Override
    public void startTraining(CbowRequestMessage message) {

    }

    @Override
    public void pickTraining(CbowRequestMessage message) {

    }

    @Override
    public void aggregationFinished(VoidAggregation aggregation) {
        // we just pick DotAggregation here

        CbowChain chain = chains.get(RequestDescriptor.createDescriptor(aggregation.getOriginatorId(), aggregation.getTaskId()));
        if (chain == null) {
            throw new RuntimeException("sI_" + transport.getShardIndex() + " Unable to find chain for specified originatorId: ["+ aggregation.getOriginatorId()+"]; taskId: [" + aggregation.getTaskId() + "]");
        }

        chain.addElement((DotAggregation) aggregation);

        finishTraining(aggregation.getOriginatorId(), aggregation.getTaskId());
    }

    @Override
    public void finishTraining(long originatorId, long taskId) {

    }


    @Override
    public String targetMessageClass() {
        return CbowRequestMessage.class.getSimpleName();
    }
}
