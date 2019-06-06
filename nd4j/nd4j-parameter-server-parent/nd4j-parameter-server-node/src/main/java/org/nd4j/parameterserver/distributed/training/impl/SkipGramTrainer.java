/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.parameterserver.distributed.training.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.logic.completion.FrameCompletionHandler;
import org.nd4j.parameterserver.distributed.logic.completion.RequestDescriptor;
import org.nd4j.parameterserver.distributed.logic.storage.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.messages.complete.FrameCompleteMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedSgDotMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.training.BaseTrainer;
import org.nd4j.parameterserver.distributed.training.chains.SkipGramChain;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

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

    protected Map<RequestDescriptor, SkipGramChain> chains = new ConcurrentHashMap<>();
    protected AtomicLong cntRounds = new AtomicLong(0);

    @Override
    public void startTraining(SkipGramRequestMessage message) {
        /**
         * All we do right HERE - is dot calculation start
         */

        /**
         * If we're on HS, we know pairs in advance: it's our points.
         */
        //        log.info("sI_{} adding SkipGramChain originator: {}; frame: {}; task: {}", transport.getShardIndex(), message.getOriginatorId(), message.getFrameId(), message.getTaskId());
        SkipGramChain chain = new SkipGramChain(message.getOriginatorId(), message.getTaskId(), message.getFrameId());
        chain.addElement(message);

        //        log.info("Starting chain [{}]", chain.getTaskId());


        chains.put(RequestDescriptor.createDescriptor(message.getOriginatorId(), message.getTaskId()), chain);

        // we assume this is HS round
        //if (message.getPoints() != null && message.getPoints().length > 0) {

        int row_syn0[] = new int[0]; //replicate(message.getW2(), message.getPoints().length);

        int row_syn1[] = message.getPoints();

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

        if (message.getPoints().length != message.getCodes().length)
            throw new RuntimeException("Mismatiching points/codes lengths here!");

        // FIXME: taskId should be real here, since it'll be used for task chain tracking
        // as result, we'll have aggregated dot as single ordered column, which might be used for gradient calculation
        DistributedSgDotMessage ddm = new DistributedSgDotMessage(message.getTaskId(), row_syn0, row_syn1,
                        message.getW1(), message.getW2(), message.getCodes(),
                        message.getCodes() != null && message.getCodes().length > 0, message.getNegSamples(),
                        (float) message.getAlpha());

        ddm.setTargetId((short) -1);
        ddm.setOriginatorId(message.getOriginatorId());


        if (voidConfiguration.getExecutionMode() == ExecutionMode.AVERAGING) {
            transport.putMessage(ddm);
        } else if (voidConfiguration.getExecutionMode() == ExecutionMode.SHARDED) {
            transport.sendMessage(ddm);
        }

        //  } //else log.info("sI_{} Skipping step: {}", transport.getShardIndex(), chain.getTaskId());

    }

    /**
     * This method will be called from non-initialized Shard context
     * @param message
     */
    @Override
    public void pickTraining(@NonNull SkipGramRequestMessage message) {
        RequestDescriptor descriptor =
                        RequestDescriptor.createDescriptor(message.getOriginatorId(), message.getTaskId());
        if (!chains.containsKey(descriptor)) {
            SkipGramChain chain = new SkipGramChain(message);
            //            log.info("sI_{} Picking chain: originator: {}; taskId: {}", transport.getShardIndex(), message.getOriginatorId(), message.getTaskId());
            chains.put(descriptor, chain);
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

        SkipGramChain chain = chains.get(
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
        SkipGramChain chain = chains.get(chainDesc);

        if (chain == null)
            throw new RuntimeException("Unable to find chain for specified taskId: [" + taskId + "]");

        SkipGramRequestMessage sgrm = chain.getRequestMessage();
        double alpha = sgrm.getAlpha();

        //log.info("Executing SkipGram round on shard_{}; taskId: {}", transport.getShardIndex(), taskId);

        // TODO: We DON'T want this code being here
        // TODO: We DO want this algorithm to be native
        INDArray expTable = storage.getArray(WordVectorStorage.EXP_TABLE);
        INDArray dots = chain.getDotAggregation().getAccumulatedResult();

        INDArray syn0 = storage.getArray(WordVectorStorage.SYN_0);
        INDArray syn1 = storage.getArray(WordVectorStorage.SYN_1);
        INDArray syn1Neg = storage.getArray(WordVectorStorage.SYN_1_NEGATIVE);

        INDArray neu1e = Nd4j.create(syn0.columns());

        int e = 0;

        boolean updated = false;

        // apply optional SkipGram HS gradients
        if (sgrm.getCodes().length > 0) {
            for (; e < sgrm.getCodes().length; e++) {
                float dot = dots.getFloat(e);

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

                updated = true;
                Nd4j.getBlasWrapper().axpy(new Double(g), syn1.getRow(sgrm.getPoints()[e]), neu1e);
                Nd4j.getBlasWrapper().axpy(new Double(g), syn0.getRow(sgrm.getW2()), syn1.getRow(sgrm.getPoints()[e]));
            }
        }

        // apply optional NegSample gradients
        if (sgrm.getNegSamples() > 0) {
            // here we assume that we already
            int cnt = 0;
            for (; e < sgrm.getNegSamples() + 1; e++, cnt++) {
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
                Nd4j.getBlasWrapper().axpy(new Double(g), syn1Neg.getRow(sgrm.getNegatives()[cnt]), neu1e);
                Nd4j.getBlasWrapper().axpy(new Double(g), syn0.getRow(sgrm.getW2()),
                                syn1Neg.getRow(sgrm.getNegatives()[cnt]));
            }
        }

        if (updated)
            Nd4j.getBlasWrapper().axpy(new Double(1.0), neu1e, syn0.getRow(sgrm.getW2()));

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
            log.info("sI_{} isn't tracking this frame: Originator: {}, frameId: {}, taskId: {}",
                            transport.getShardIndex(), chain.getOriginatorId(), chain.getFrameId(), taskId);
        }

        if (cntRounds.incrementAndGet() % 100000 == 0)
            log.info("{} training rounds finished...", cntRounds.get());

        // don't forget to remove chain, it'll become a leak otherwise
        chains.remove(chainDesc);
    }

}
