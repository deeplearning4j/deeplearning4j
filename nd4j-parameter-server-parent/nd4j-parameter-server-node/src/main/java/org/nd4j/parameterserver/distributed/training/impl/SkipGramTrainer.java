package org.nd4j.parameterserver.distributed.training.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
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
@Slf4j
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
        SkipGramChain chain = new SkipGramChain(message.getTaskId());
        chain.addElement(message);

        log.info("Starting chain [{}]", chain.getTaskId());


        chains.put(chain.getTaskId(), chain);

        if (message.getPoints() != null && message.getPoints().length > 0) {
            // we assume this is HS round
            int row_syn0[] = replicate(message.getW1(), message.getPoints().length);

            // FIXME: taskId should be real here, since it'll be used for task chain tracking
            // as result, we'll have aggregated dot as single ordered column, which might be used for gradient calculation
            DistributedDotMessage ddm = new DistributedDotMessage(message.getTaskId(), WordVectorStorage.SYN_0, WordVectorStorage.SYN_1, row_syn0, message.getPoints(),
                    message.getW1(),
                    message.getW2(),
                    message.getCodes(),
                    message.getPoints() != null && message.getPoints().length > 0,
                    message.getNegSamples(),
                    (float) message.getAlpha());

            transport.sendMessage(ddm);
        }

        // negSampling round
        if (message.getNegSamples() > 0) {

        }
    }

    /**
     * This method will be called from non-initialized Shard context
     * @param message
     */
    @Override
    public void pickTraining(@NonNull SkipGramRequestMessage message) {
        if (!chains.containsKey(message.getTaskId())) {
            SkipGramChain chain = new SkipGramChain(message);
            chains.put(chain.getTaskId(), chain);
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
    public void aggregationFinished(@NonNull VoidAggregation aggregation) {
        // the only possible aggregation here is DotAggregation, actually
        // so we just calculate gradients here

        SkipGramChain chain = chains.get(aggregation.getTaskId());

        if (chain == null) {
            throw new RuntimeException("Unable to find chain for specified taskId: [" + aggregation.getTaskId() + "]");
        }

        chain.addElement((DotAggregation) aggregation);

        finishTraining(aggregation.getTaskId());
    }

    @Override
    public void finishTraining(long taskId) {
        // TODO: real values needed here
        SkipGramChain chain = chains.get(taskId);

        if (chain == null)
            throw new RuntimeException("Unable to find chain for specified taskId: [" + taskId + "]");

        SkipGramRequestMessage sgrm = chain.getRequestMessage();
        double alpha = sgrm.getAlpha();

        log.info("Executing SkipGram round on shard_{}", transport.getShardIndex());

        // TODO: We DON'T want this code being here
        // TODO: We DO want this algorithm to be native
        INDArray expTable = storage.getArray(WordVectorStorage.EXP_TABLE);
        INDArray dots = chain.getDotAggregation().getAccumulatedResult();


        INDArray syn0 = storage.getArray(WordVectorStorage.SYN_0);
        INDArray syn1 = storage.getArray(WordVectorStorage.SYN_1);

        INDArray neu1e = Nd4j.create(syn1.columns());

        for (int e = 0; e < dots.length(); e++) {
            float dot = dots.getFloat(e);

            log.info("dot at shard_{}: [{}]", transport.getShardIndex(), dot);

            if (dot < -HS_MAX_EXP || dot >= HS_MAX_EXP) {
                continue;
            }

            int idx = (int) ((dot + HS_MAX_EXP) * ((float) expTable.length() / HS_MAX_EXP / 2.0));

            if (idx >= expTable.length() || idx < 0) {
                continue;
            }

            int code = chain.getRequestMessage().getCodes()[e];
            double f = expTable.getFloat(idx);
            double g = (1 - code - f) * alpha;

            log.info("gradient at shard_{}: [{}]", transport.getShardIndex(), g);

            // FIXME: this is wrong, just a draft showing an idea
            Nd4j.getBlasWrapper().axpy(new Double(g), syn1.getRow(sgrm.getPoints()[e]), neu1e );
            Nd4j.getBlasWrapper().axpy(new Double(g), syn0.getRow(sgrm.getW1()), syn1.getRow(sgrm.getPoints()[e]));
        }

        Nd4j.getBlasWrapper().axpy(new Double(1.0), neu1e, syn0.getRow(sgrm.getW1()));
    }
}
