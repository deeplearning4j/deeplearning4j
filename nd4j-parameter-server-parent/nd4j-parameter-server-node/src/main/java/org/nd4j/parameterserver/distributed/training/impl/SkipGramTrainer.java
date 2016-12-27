package org.nd4j.parameterserver.distributed.training.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.logic.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.Chain;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.aggregations.VoidAggregation;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedDotMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.training.BaseTrainer;
import org.nd4j.parameterserver.distributed.training.chains.SkipGramChain;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Distributed SkipGram trainer
 *
 * TrainingDriver idea is simple:
 *      1) We get request from Client
 *      2) We initiate training by issuing DotRequest
 *      3) Each Shard does Dot accumulation
 *      4) As soon as Dot aggregated, we calculate gradients independently
 *      5) As soon as they are ready - we just apply them to appropriate
 *
 * @author raver119@gmail.com
 */
public class SkipGramTrainer extends BaseTrainer<SkipGramRequestMessage> {
    private static final float HS_MAX_EXP = 6.0f;

    protected Map<Long, SkipGramChain> chains = new ConcurrentHashMap<>();

    @Override
    public void startTraining(SkipGramRequestMessage message) {
        /**
         * All we do right HERE - is dot calculation start
         */

        /**
         * If we're on HS, we know pairs in advance: it's our points.
         */
        SkipGramChain chain = new SkipGramChain(0L);
        chain.addElement(message);

        chains.put(chain.getTaskId(), chain);

        if (message.getPoints() != null && message.getPoints().length > 0) {
            // we assume this is HS round
            int row_syn0[] = replicate(message.getW1(), message.getPoints().length);

            // FIXME: taskId should be real here, since it'll be used for task chain tracking
            // as result, we'll have aggregated dot as single ordered column, which might be used for gradient calculation
            DistributedDotMessage ddm = new DistributedDotMessage(0L, WordVectorStorage.SYN_0, WordVectorStorage.SYN_1, row_syn0, message.getPoints());
            transport.sendMessage(ddm);
        }

        // negSampling round
        if (message.getNegSamples() > 0) {

        }
    }

    @Override
    public String targetMessageClass() {
        return SkipGramRequestMessage.class.getSimpleName();
    }

    /**
     * This method is invoked after particular aggregation finished
     * @param aggregation
     */
    @Override
    public void aggregationFinished(VoidAggregation aggregation) {
        // the only possible aggregation here is DotAggregation, actually
        // so we just calculate gradients here

        SkipGramChain chain = chains.get(aggregation.getTaskId());
        chain.addElement((DotAggregation) aggregation);

        finishTraining(aggregation.getTaskId());
    }

    @Override
    public void finishTraining(long taskId) {
        // TODO: real values needed here
        SkipGramChain chain = chains.get(taskId);

        if (chain == null)
            throw new RuntimeException("Unable to find chain for specified taskId: [" + taskId + "]");

        float alpha = chain.getRequestMessage().getAlpha();
        int code = 0;

        // TODO: We DON'T want this code being here
        // TODO: We want algorithm below to be native
        INDArray expTable = storage.getArray(WordVectorStorage.EXP_TABLE);
        INDArray dots = chain.getDotAggregation().getAccumulatedResult();

        for (int e = 0; e < dots.length(); e++) {
            float dot = dots.getFloat(e);

            if (dot < -HS_MAX_EXP || dot >= HS_MAX_EXP) {
                dots.putScalar(e, Float.NaN);
                continue;
            }

            int idx = (int) ((dot + HS_MAX_EXP) * ((float) expTable.length() / HS_MAX_EXP / 2.0));

            if (idx >= expTable.length() || idx < 0) {
                dots.putScalar(e, Float.NaN);
                continue;
            }

            float f = expTable.getFloat(idx);
            float g = (1 - code - f) * alpha;
            dots.putScalar(e, g);
        }
    }
}
