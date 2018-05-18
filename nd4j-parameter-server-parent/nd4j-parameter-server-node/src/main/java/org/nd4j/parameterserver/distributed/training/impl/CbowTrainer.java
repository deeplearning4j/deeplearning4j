package org.nd4j.parameterserver.distributed.training.impl;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.logic.completion.FrameCompletionHandler;
import org.nd4j.parameterserver.distributed.logic.completion.RequestDescriptor;
import org.nd4j.parameterserver.distributed.logic.storage.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.complete.FrameCompleteMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedCbowDotMessage;
import org.nd4j.parameterserver.distributed.messages.requests.CbowRequestMessage;
import org.nd4j.parameterserver.distributed.training.BaseTrainer;
import org.nd4j.parameterserver.distributed.training.chains.CbowChain;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class CbowTrainer extends BaseTrainer<CbowRequestMessage> {
    private static final float HS_MAX_EXP = 6.0f;

    protected Map<RequestDescriptor, CbowChain> chains = new ConcurrentHashMap<>();
    protected AtomicLong cntRounds = new AtomicLong(0);



    @Override
    public void startTraining(CbowRequestMessage message) {
        CbowChain chain = new CbowChain(message);
        chain.addElement(message);

        chains.put(RequestDescriptor.createDescriptor(message.getOriginatorId(), message.getTaskId()), chain);

        int row_syn1[] = message.getSyn1rows();

        if (message.getNegSamples() > 0) {
            int rows = (int) storage.getArray(WordVectorStorage.SYN_0).rows();
            int tempArray[] = new int[message.getNegSamples() + 1];
            tempArray[0] = message.getW1();

            for (int e = 1; e < message.getNegSamples() + 1; e++) {
                while (true) {
                    int rnd = RandomUtils.nextInt(0, rows);
                    if (rnd != message.getW1()) {
                        tempArray[e] = rnd;
                        break;
                    }
                }
            }

            row_syn1 = ArrayUtils.addAll(row_syn1, tempArray);

            message.setNegatives(tempArray);
        }

        if (message.getSyn0rows() == null || message.getSyn0rows().length < 1)
            throw new RuntimeException("Empty syn0rows!");

        DistributedCbowDotMessage dcdm = new DistributedCbowDotMessage(message.getTaskId(), message.getSyn0rows(),
                        row_syn1, message.getW1(), message.getCodes(), message.getCodes().length > 0,
                        (short) message.getNegSamples(), (float) message.getAlpha());
        dcdm.setTargetId((short) -1);
        dcdm.setOriginatorId(message.getOriginatorId());

        if (voidConfiguration.getExecutionMode() == ExecutionMode.AVERAGING) {
            transport.putMessage(dcdm);
        } else if (voidConfiguration.getExecutionMode() == ExecutionMode.SHARDED) {
            transport.sendMessage(dcdm);
        }
    }

    @Override
    public void pickTraining(CbowRequestMessage message) {
        RequestDescriptor descriptor =
                        RequestDescriptor.createDescriptor(message.getOriginatorId(), message.getTaskId());
        if (!chains.containsKey(descriptor)) {
            CbowChain chain = new CbowChain(message);
            chain.addElement(message);
            chains.put(descriptor, chain);
        }
    }

    @Override
    public void aggregationFinished(VoidAggregation aggregation) {
        // we just pick DotAggregation here

        CbowChain chain = chains.get(
                        RequestDescriptor.createDescriptor(aggregation.getOriginatorId(), aggregation.getTaskId()));
        if (chain == null) {
            throw new RuntimeException("sI_" + transport.getShardIndex()
                            + " Unable to find chain for specified originatorId: [" + aggregation.getOriginatorId()
                            + "]; taskId: [" + aggregation.getTaskId() + "]");
        }

        chain.addElement((DotAggregation) aggregation);

        finishTraining(aggregation.getOriginatorId(), aggregation.getTaskId());
    }

    @Override
    public void finishTraining(long originatorId, long taskId) {
        RequestDescriptor chainDesc = RequestDescriptor.createDescriptor(originatorId, taskId);
        CbowChain chain = chains.get(chainDesc);

        if (chain == null)
            throw new RuntimeException("Unable to find chain for specified taskId: [" + taskId + "]");

        CbowRequestMessage cbr = chain.getCbowRequest();
        double alpha = cbr.getAlpha();

        //log.info("Executing SkipGram round on shard_{}; taskId: {}", transport.getShardIndex(), taskId);

        // TODO: We DON'T want this code being here
        // TODO: We DO want this algorithm to be native
        INDArray expTable = storage.getArray(WordVectorStorage.EXP_TABLE);
        INDArray dots = chain.getDotAggregation().getAccumulatedResult();

        INDArray syn0 = storage.getArray(WordVectorStorage.SYN_0);
        INDArray syn1 = storage.getArray(WordVectorStorage.SYN_1);
        INDArray syn1Neg = storage.getArray(WordVectorStorage.SYN_1_NEGATIVE);

        INDArray words = Nd4j.pullRows(storage.getArray(WordVectorStorage.SYN_0), 1, cbr.getSyn0rows(), 'c');
        INDArray neue = words.mean(0);

        INDArray neu1e = Nd4j.create(syn0.columns());

        int e = 0;

        boolean updated = false;

        // probably applying HS part
        if (cbr.getCodes().length > 0) {
            for (; e < cbr.getCodes().length; e++) {
                float dot = dots.getFloat(e);

                if (dot < -HS_MAX_EXP || dot >= HS_MAX_EXP) {
                    continue;
                }

                int idx = (int) ((dot + HS_MAX_EXP) * ((float) expTable.length() / HS_MAX_EXP / 2.0));

                if (idx >= expTable.length() || idx < 0) {
                    continue;
                }

                int code = cbr.getCodes()[e];
                double f = expTable.getFloat(idx);
                double g = (1 - code - f) * alpha;

                updated = true;
                Nd4j.getBlasWrapper().axpy(new Double(g), syn1.getRow(cbr.getSyn1rows()[e]), neu1e);
                Nd4j.getBlasWrapper().axpy(new Double(g), neue, syn1.getRow(cbr.getSyn1rows()[e]));
            }
        }

        if (cbr.getNegSamples() > 0) {
            int cnt = 0;
            for (; e < cbr.getNegSamples() + 1; e++, cnt++) {
                float dot = dots.getFloat(e);

                float code = cnt == 0 ? 1.0f : 0.0f;
                double g = 0.0f;

                if (dot > HS_MAX_EXP)
                    g = (code - 1) * alpha;
                else if (dot < -HS_MAX_EXP)
                    g = (code - 0) * alpha;
                else {
                    int idx = (int) ((dot + HS_MAX_EXP) * (expTable.length() / HS_MAX_EXP / 2.0));
                    if (idx >= expTable.length() || idx < 0)
                        continue;

                    g = (code - expTable.getDouble(idx)) * alpha;
                }

                updated = true;
                Nd4j.getBlasWrapper().axpy(new Double(g), syn1Neg.getRow(cbr.getNegatives()[cnt]), neu1e);
                Nd4j.getBlasWrapper().axpy(new Double(g), neue, syn1Neg.getRow(cbr.getNegatives()[cnt]));
            }
        }

        if (updated)
            for (int i = 0; i < cbr.getSyn0rows().length; i++) {
                Nd4j.getBlasWrapper().axpy(new Double(1.0), neu1e, syn0.getRow(cbr.getSyn0rows()[i]));
            }

        // we send back confirmation message only from Shard which received this message
        RequestDescriptor descriptor = RequestDescriptor.createDescriptor(chain.getOriginatorId(), chain.getFrameId());

        if (completionHandler.isTrackingFrame(descriptor)) {
            completionHandler.notifyFrame(chain.getOriginatorId(), chain.getFrameId(), chain.getTaskId());

            if (completionHandler.isCompleted(descriptor)) {
                FrameCompletionHandler.FrameDescriptor frameDescriptor =
                                completionHandler.getCompletedFrameInfo(descriptor);


                // TODO: there is possible race condition here
                if (frameDescriptor != null) {
                    FrameCompleteMessage fcm = new FrameCompleteMessage(chain.getFrameId());
                    fcm.setOriginatorId(frameDescriptor.getFrameOriginatorId());
                    transport.sendMessage(fcm);
                } else {
                    log.warn("Frame double spending detected");
                }
            }
        } else {
            //log.info("sI_{} isn't tracking this frame: Originator: {}, frameId: {}, taskId: {}", transport.getShardIndex(), chain.getOriginatorId(), chain.getFrameId(), taskId );
        }



        if (cntRounds.incrementAndGet() % 100000 == 0)
            log.info("{} training rounds finished...", cntRounds.get());

        // don't forget to remove chain, it'll become a leak otherwise
        chains.remove(chainDesc);
    }


    @Override
    public String targetMessageClass() {
        return CbowRequestMessage.class.getSimpleName();
    }
}
